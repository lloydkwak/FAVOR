"""
LIBERO Phase 1 fault sweep: B1 (no intervention) only, across
sweep_grid_libero.py's 7 joints x 1 fault condition (locked) x 3 tasks =
21 conditions, N_TEST=20 episodes each, seeded (TEST_START_SEED fixed
across all conditions for valid paired comparison later against a
selection-based method once implemented).

One process per condition would deadlock-avoid perfectly but cost a
container spin-up each time; instead this runs all conditions for one
task within a single process (task-level env/policy reuse), and expects
to be invoked once per task from the shell -- matching the one-process-
per-environment-family pattern that has worked reliably (as opposed to
constructing environments for DIFFERENT tasks in one process, which
deadlocked repeatedly this session).

Usage: python run_libero_fault_sweep_phase1.py <task_name>
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid_libero import JOINTS, FAULT_CONDITIONS, N_TEST, TEST_START_SEED, TASKS

OUT_DIR = "/workspace/results/libero_fault_sweep_phase1"
os.makedirs(OUT_DIR, exist_ok=True)

task_name = sys.argv[1]
assert task_name in TASKS, f"unknown task {task_name!r}, expected one of {list(TASKS)}"

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_sweep_scratch_{task_name}")
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
            output_dir=f"/workspace/results/_sweep_run_{task_name}",
            dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
            n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key="agentview_image", abs_action=True,
            actuation_mode="joint", joint_kp=150,
        )
        policy = NativeJointPolicy(base_policy, base_seed=42, fault_spec=None)
        log = runner.run(policy)
        score = log.get("test/mean_score")
        per_episode = {k.replace("test/sim_max_reward_", ""): v
                       for k, v in log.items() if k.startswith("test/sim_max_reward_")}
        print(f"  {task_name}/{joint_name}/{fault_type}/{severity}/b1: {score}")

        with open(out_path, "w") as f:
            json.dump({"b1": score, "per_episode": per_episode}, f, indent=2)
        print(f"SAVED: {fname}")

print(f"TASK {task_name} SWEEP COMPLETE")
