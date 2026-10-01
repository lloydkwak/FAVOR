"""
Fault-knowledge misspecification sweep for Priority IK (lam2=0.2).
Usage: python run_libero_fault_sweep_ms.py <tag> <task> locked
       python run_libero_fault_sweep_ms.py <tag> <task> range <level_idx>
tag: rs050 rs150 (range window x0.5 / x1.5), lop010 lom010 (lock angle +/-0.1 rad),
     wj (wrong neighbour joint), online (detect from tracking error, no prior knowledge)
Joints: j1 j3 j5 j6 j7 (j2/j4 excluded: every method ~0).
Output: results/libero_fault_sweep_{locked|range}_ms_<tag>/<same file names as other sweeps>
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import favor_fault_runner  # noqa: F401
from favor_fault_runner import FaultRobomimicImageRunner
from native_joint_policy_ms import MisspecPrioPolicy, summarize_ms_log

TAGS = {"rs050": ("range_scale", 0.5), "rs150": ("range_scale", 1.5),
        "lop010": ("lock_offset", 0.1), "lom010": ("lock_offset", -0.1),
        "lop000": ("lock_offset", 0.0), "lop002": ("lock_offset", 0.02), "lop005": ("lock_offset", 0.05),
        "wj": ("wrong_joint", None), "online": ("online", None), "online2": ("online", None), "online3": ("online", None), "online4": ("online", None), "delay1": ("delay", 1)}
DET_KW = {"online2": {"det_tau": 0.02, "det_rel": 3.0, "det_k": 2},
          "online3": {"det_mode": "stuck", "det_k": 1},
          "online4": {"det_mode": "stuck", "det_k": 1, "det_margin": 0.0}}
TAGS["lop001"] = ("lock_offset", 0.01)
USE_JOINTS = {f"robot0_joint{i}" for i in [int(x) for x in os.environ.get("MS_JOINTS", "1,3,5,6,7").split(",")]}
tag, task_name, fault = sys.argv[1], sys.argv[2], sys.argv[3]
assert tag in TAGS, tag
assert fault in ("locked", "range"), fault
ms_mode, ms_param = TAGS[tag]
method = f"ms_{tag}"

if fault == "locked":
    from sweep_grid_libero import JOINTS, FAULT_CONDITIONS, N_TEST, TEST_START_SEED, TASKS
    conds = [(j, ft, sev, None) for j in JOINTS for ft, sev in FAULT_CONDITIONS if ft == "locked"]
    OUT_DIR = f"/workspace/results/libero_fault_sweep_locked_{method}"
else:
    from sweep_grid_libero_range import (JOINTS, LEVELS, FAULT_TYPE, N_TEST, TEST_START_SEED,
                                         TASKS, severity_for)
    lvl_name, keep = LEVELS[int(sys.argv[4])]
    conds = [(j, FAULT_TYPE, float(severity_for(task_name, j, keep)), lvl_name) for j in JOINTS]
    OUT_DIR = f"/workspace/results/libero_fault_sweep_range_{method}"
conds = [c for c in conds if c[0] in USE_JOINTS]
assert task_name in TASKS, task_name
os.makedirs(OUT_DIR, exist_ok=True)

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=f"/workspace/results/_sweep_scratch_{method}_{fault}_{task_name}")
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to(torch.device("cuda:0")); base_policy.eval()

for joint_name, fault_type, severity, lvl in conds:
    if lvl is None:
        fname = f"{task_name}_{joint_name}_{fault_type}_{severity}.json".replace("None", "na")
        ftag = f"{task_name}/{joint_name}/locked/{method}"
    else:
        fname = f"{task_name}_{joint_name}_{fault_type}_{lvl}.json"
        ftag = f"{task_name}/{joint_name}/{lvl}(sev={severity:.4f})/{method}"
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"\nSKIP (exists): {fname}", flush=True); continue
    runner = FaultRobomimicImageRunner(
        output_dir=f"/workspace/results/_sweep_run_{method}_{fault}_{task_name}",
        dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
        max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        render_obs_key="agentview_image", abs_action=True,
        actuation_mode="joint", joint_kp=TASKS[task_name].get("joint_kp", 150),
    )
    policy = MisspecPrioPolicy(base_policy, ms_mode, ms_param, lam2=0.2, base_seed=42, **DET_KW.get(tag, {}),
                               env_ref=runner.env, fault_joint_name=joint_name,
                               fault_type=fault_type, fault_severity=severity)
    log = runner.run(policy)
    score = log.get("test/mean_score")
    per_episode = {k.replace("test/sim_max_reward_", ""): v
                   for k, v in log.items() if k.startswith("test/sim_max_reward_")}
    try:
        stats = summarize_ms_log(policy)
    except Exception as e:
        stats = {"summary_error": str(e)}
    with open(out_path, "w") as f:
        json.dump({method: score, "per_episode": per_episode, "task": task_name, "joint": joint_name,
                   "fault_type": fault_type, "severity": severity, "level": lvl, "n_test": N_TEST,
                   "ms_mode": ms_mode, "ms_param": ms_param, "stats": stats}, f, indent=2)
    print(f"\nRESULT {ftag}: {score}  stats={stats}", flush=True)
print(f"\nDONE {task_name} {fault} {method}", flush=True)
