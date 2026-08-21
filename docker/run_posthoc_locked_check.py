"""
Posthoc (single terminal projection, no renoise) vs eci comparison, for
Can/Lift, locked fault only, all 7 joints. Purpose: confirms whether
joint2/joint4's exact-zero result (unaffected by n_resample increases in
eci mode) is also unaffected under posthoc -- if BOTH mechanisms show
identically zero, this further supports the "training-distribution support
gap" hypothesis (no correction mechanism, regardless of sophistication,
can recover a region the network never learned), since posthoc and eci
are structurally very different interventions (one-shot vs iterated).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid import JOINTS

TASKS = {
    "can": {
        "ckpt": "/workspace/data/outputs/joint_train_can_run/checkpoints/epoch=0165-test_mean_score=1.000.ckpt",
        "dataset": "/workspace/diffusion_policy/data/robomimic/datasets/can/ph/image_abs.hdf5",
    },
    "lift": {
        "ckpt": "/workspace/data/outputs/joint_train_lift_run/checkpoints/epoch=0055-test_mean_score=1.000.ckpt",
        "dataset": "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5",
    },
}
N_TEST = 50

for task_name, info in TASKS.items():
    payload = torch.load(open(info["ckpt"], 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_scratch_posthoc_{task_name}")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    base_policy.to(torch.device("cuda:0"))
    base_policy.eval()

    for joint_name in JOINTS:
        runner = FaultRobomimicImageRunner(
            output_dir=f"/workspace/results/_posthoc_{task_name}_{joint_name}",
            dataset_path=info["dataset"], shape_meta=cfg.task.shape_meta,
            fault_joint_name=joint_name, fault_type="locked", fault_severity=None,
            n_train=0, n_test=N_TEST, test_start_seed=10000, n_envs=25,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key='agentview_image', abs_action=True,
            actuation_mode='joint',
        )
        policy = NativeJointPolicy(
            base_policy, mode='posthoc', base_seed=42, env_ref=runner.env,
            fault_joint_name=joint_name, fault_type="locked", fault_severity=None)
        log = runner.run(policy)
        score = log.get("test/mean_score")
        print(f"{task_name} / {joint_name} / posthoc(locked): score = {score}")
