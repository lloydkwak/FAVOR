"""
range_reduced fault sweep (B1 / E-C-I / Select / Random-N), one process per
(task, level, method) -- one task per process, as in the locked sweeps.
Severity is set per (task, joint) from the demos (sweep_grid_libero_range.py).
Policy construction is identical to run_libero_confirm_n50.py.
Resumable: result files that already exist are SKIPPED.

Usage: python run_libero_fault_sweep_range.py <task> <level_idx> <method>
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from ik_redistribution import summarize_ik_log
from sweep_grid_libero_range import (JOINTS, LEVELS, FAULT_TYPE, N_TEST, TEST_START_SEED,
                                     TASKS, severity_for, demo_excursion)

task_name, level_idx, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert task_name in TASKS, f"unknown task {task_name!r}, expected one of {list(TASKS)}"
assert mode in ("b1", "eci", "select", "random_n", "ik", "ik_pose"), mode
level, keep_frac = LEVELS[level_idx]

OUT_DIR = f"/workspace/results/libero_fault_sweep_range_{mode}"
os.makedirs(OUT_DIR, exist_ok=True)

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=f"/workspace/results/_sweep_scratch_range_{mode}_{task_name}")
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

for joint_name in JOINTS:
    j = int(joint_name.replace("robot0_joint", "")) - 1
    severity = severity_for(task_name, joint_name, keep_frac)
    fname = f"{task_name}_{joint_name}_{FAULT_TYPE}_{level}.json"
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"\nSKIP (exists): {mode}/{fname}", flush=True)
        continue

    runner = FaultRobomimicImageRunner(
        output_dir=f"/workspace/results/_sweep_run_range_{mode}_{task_name}",
        dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name, fault_type=FAULT_TYPE, fault_severity=severity,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
        max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        render_obs_key="agentview_image", abs_action=True,
        actuation_mode="joint", joint_kp=TASKS[task_name].get("joint_kp", 150),
    )
    if mode == "b1":
        policy = NativeJointPolicy(base_policy, base_seed=42, fault_spec=None)
    else:
        kw = dict(base_seed=42, mode=("ik" if mode == "ik_pose" else mode), env_ref=runner.env,
                  fault_joint_name=joint_name, fault_type=FAULT_TYPE, fault_severity=severity)
        if mode in ("select", "random_n"):
            kw.update(n_select=32, chunk_size=8)
        if mode == "ik_pose":
            kw.update(ik_overrides={"rot_weight": 1.0})
        policy = NativeJointPolicy(base_policy, **kw)

    log = runner.run(policy)
    score = log.get("test/mean_score")
    per_episode = {k.replace("test/sim_max_reward_", ""): v
                   for k, v in log.items() if k.startswith("test/sim_max_reward_")}
    with open(out_path, "w") as f:
        json.dump({mode: score, "per_episode": per_episode, "task": task_name,
                   "joint": joint_name, "fault_type": FAULT_TYPE, "level": level,
                   "keep_frac": keep_frac, "severity": severity,
                   "demo_excursion_rad": float(demo_excursion(task_name)[j]),
                   "n_test": N_TEST, "ik_stats": summarize_ik_log(policy)}, f, indent=2)
    print(f"\nRESULT {task_name}/{joint_name}/{level}(sev={severity:.4f})/{mode}: {score}", flush=True)

print(f"\nDONE {task_name} {level} {mode}", flush=True)
