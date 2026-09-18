"""
Select (N=32, chunk_size=8) sweep for range_reduced faults, mirroring
run_libero_fault_sweep_phase2_select.py exactly except for grid source
and output dir. Same n_envs=5 (measured to hold peak GPU memory flat
regardless of n_envs at chunk_size=8, see prior commit messages).

fault_certificate.py's apply_fault() already handles range_reduced (clamp
to [q_lo, q_hi], distinct from locked's fixed q_lock), and
native_joint_policy.py's select branch already reads fault_type generically
from self.fault_type -- no code changes needed here, only a new grid.

Usage: python run_libero_fault_sweep_range_select.py <task_name>
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid_libero_range import JOINTS, FAULT_CONDITIONS, N_TEST, TEST_START_SEED, TASKS

OUT_DIR = "/workspace/results/libero_fault_sweep_range_select"
os.makedirs(OUT_DIR, exist_ok=True)

task_name = sys.argv[1]
assert task_name in TASKS, f"unknown task {task_name!r}, expected one of {list(TASKS)}"

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_sweep_scratch_range_select_{task_name}")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

for joint_name in JOINTS:
    for fault_type, severity in FAULT_CONDITIONS:
        fname = f"{task_name}_{joint_name}_{fault_type}_{severity}.json".replace("None", "na")
        out_path = os.path.join(OUT_DIR, fname)
        if os.path.exists(out_path):
            print(f"SKIP (exists): {fname}")
            continue

        runner = FaultRobomimicImageRunner(
            output_dir=f"/workspace/results/_sweep_run_range_select_{task_name}",
            dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
            n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key="agentview_image", abs_action=True,
            actuation_mode="joint", joint_kp=150,
        )
        policy = NativeJointPolicy(
            base_policy, base_seed=42, mode='select', env_ref=runner.env,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
            n_select=32, chunk_size=8,
        )
        log = runner.run(policy)
        score = log.get("test/mean_score")
        per_episode = {k.replace("test/sim_max_reward_", ""): v
                       for k, v in log.items() if k.startswith("test/sim_max_reward_")}
        print(f"  {task_name}/{joint_name}/{fault_type}/{severity}/select: {score}")

        with open(out_path, "w") as f:
            json.dump({"select": score, "per_episode": per_episode}, f, indent=2)
        print(f"SAVED: {fname}")

print(f"TASK {task_name} RANGE SELECT SWEEP COMPLETE")
