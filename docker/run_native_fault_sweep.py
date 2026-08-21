"""
Fault sweep for native joint-space checkpoints: B1 vs FAVOR-eci vs
FAVOR-posthoc, across sweep_grid.py's 7 joints x 5 fault conditions x 3
tasks. Paired comparison (same env seeds, N_TEST episodes) per condition,
resumable (skips conditions with an existing result file).
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid import JOINTS, FAULT_CONDITIONS, N_TEST, TEST_START_SEED

NATIVE_TASKS = {
    "square": "/workspace/data/outputs/2026.08.10/05.53.47_train_diffusion_unet_hybrid_square_image_joint/checkpoints/latest.ckpt",
    "can": "/workspace/data/outputs/joint_train_can_run/checkpoints/epoch=0165-test_mean_score=1.000.ckpt",
    "lift": "/workspace/data/outputs/joint_train_lift_run/checkpoints/epoch=0055-test_mean_score=1.000.ckpt",
}
DATASETS = {
    "square": "/workspace/diffusion_policy/data/robomimic/datasets/square/ph/image_abs.hdf5",
    "can": "/workspace/diffusion_policy/data/robomimic/datasets/can/ph/image_abs.hdf5",
    "lift": "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5",
}
OUT_DIR = "/workspace/results/native_fault_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

_policy_cache = {}
def get_base_policy(task_name):
    if task_name not in _policy_cache:
        payload = torch.load(open(NATIVE_TASKS[task_name], 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_sweep_scratch_{task_name}")
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.to(torch.device("cuda:0"))
        policy.eval()
        _policy_cache[task_name] = (policy, cfg)
    return _policy_cache[task_name]


def run_condition(task_name, joint_name, fault_type, severity):
    fname = f"{task_name}_{joint_name}_{fault_type}_{severity}.json".replace("None", "na")
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"SKIP (exists): {fname}")
        return

    base_policy, cfg = get_base_policy(task_name)
    results = {}
    for label, policy_kwargs in [
        ("b1", dict(fault_spec=None)),
        ("favor_eci", dict(mode='eci')),
        # favor_posthoc temporarily excluded from the sweep (per instruction)
        # to focus compute on the B1 vs FAVOR-eci comparison first.
    ]:
        runner = FaultRobomimicImageRunner(
            output_dir=f"/workspace/results/_sweep_run_{task_name}",
            dataset_path=DATASETS[task_name], shape_meta=cfg.task.shape_meta,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
            n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=25,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key='agentview_image', abs_action=True,
            actuation_mode='joint',
        )
        if label == "b1":
            policy = NativeJointPolicy(base_policy, base_seed=42, **policy_kwargs)
        else:
            policy = NativeJointPolicy(
                base_policy, base_seed=42, env_ref=runner.env,
                fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
                **policy_kwargs)
        log = runner.run(policy)
        results[label] = log.get("test/mean_score")
        print(f"  {task_name}/{joint_name}/{fault_type}/{severity}/{label}: {results[label]}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"SAVED: {fname}")


if __name__ == "__main__":
    for task_name in NATIVE_TASKS:
        for joint_name in JOINTS:
            for fault_type, severity in FAULT_CONDITIONS:
                run_condition(task_name, joint_name, fault_type, severity)
    print("SWEEP COMPLETE")
