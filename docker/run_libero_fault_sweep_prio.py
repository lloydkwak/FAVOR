"""
Priority-IK sweep (PrioIKPolicy, lam2=0.2). Same grids/seeds/policy construction as
run_libero_fault_sweep_locked.py and run_libero_fault_sweep_range.py.
Usage: python run_libero_fault_sweep_prio.py <task> locked
       python run_libero_fault_sweep_prio.py <task> range <level_idx>
Output: results/libero_fault_sweep_locked_prio/<task>_<joint>_locked_na.json
        results/libero_fault_sweep_range_prio/<task>_<joint>_range_reduced_<level>.json
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import favor_fault_runner  # noqa: F401
from favor_fault_runner import FaultRobomimicImageRunner
from native_joint_policy_prio import PrioIKPolicy, summarize_prio_log

LAM2 = 0.2
task_name, fault = sys.argv[1], sys.argv[2]
assert fault in ("locked", "range"), fault
if fault == "locked":
    from sweep_grid_libero import JOINTS, FAULT_CONDITIONS, N_TEST, TEST_START_SEED, TASKS
    conds = [(j, ft, sev, None) for j in JOINTS for ft, sev in FAULT_CONDITIONS if ft == "locked"]
    OUT_DIR = "/workspace/results/libero_fault_sweep_locked_prio"
else:
    from sweep_grid_libero_range import (JOINTS, LEVELS, FAULT_TYPE, N_TEST, TEST_START_SEED,
                                         TASKS, severity_for)
    lvl_name, keep = LEVELS[int(sys.argv[3])]
    conds = [(j, FAULT_TYPE, float(severity_for(task_name, j, keep)), lvl_name) for j in JOINTS]
    OUT_DIR = "/workspace/results/libero_fault_sweep_range_prio"
assert task_name in TASKS, task_name
os.makedirs(OUT_DIR, exist_ok=True)

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=f"/workspace/results/_sweep_scratch_prio_{fault}_{task_name}")
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to(torch.device("cuda:0")); base_policy.eval()

for joint_name, fault_type, severity, lvl in conds:
    if lvl is None:
        fname = f"{task_name}_{joint_name}_{fault_type}_{severity}.json".replace("None", "na")
        tag = f"{task_name}/{joint_name}/locked/prio"
    else:
        fname = f"{task_name}_{joint_name}_{fault_type}_{lvl}.json"
        tag = f"{task_name}/{joint_name}/{lvl}(sev={severity:.4f})/prio"
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"\nSKIP (exists): {fname}", flush=True); continue
    runner = FaultRobomimicImageRunner(
        output_dir=f"/workspace/results/_sweep_run_prio_{fault}_{task_name}",
        dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
        max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        render_obs_key="agentview_image", abs_action=True,
        actuation_mode="joint", joint_kp=TASKS[task_name].get("joint_kp", 150),
    )
    policy = PrioIKPolicy(base_policy, lam2=LAM2, base_seed=42, env_ref=runner.env,
                          fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity)
    log = runner.run(policy)
    score = log.get("test/mean_score")
    per_episode = {k.replace("test/sim_max_reward_", ""): v
                   for k, v in log.items() if k.startswith("test/sim_max_reward_")}
    with open(out_path, "w") as f:
        json.dump({"prio": score, "per_episode": per_episode, "task": task_name, "joint": joint_name,
                   "fault_type": fault_type, "severity": severity, "level": lvl, "n_test": N_TEST,
                   "lam2": LAM2, "prio_stats": summarize_prio_log(policy)}, f, indent=2)
    print(f"\nRESULT {tag}: {score}", flush=True)
print(f"\nDONE {task_name} {fault} prio", flush=True)
