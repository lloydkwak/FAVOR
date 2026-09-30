"""
Locked-fault sweep for methods added after the original locked sweeps
(e.g. 'ik'). Same grid and seeds as sweep_grid_libero.py (n_test=20,
TEST_START_SEED=10000), same policy construction as the range sweep.
Output: results/libero_fault_sweep_locked_<method>/<task>_<joint>_locked_na.json
Usage: python run_libero_fault_sweep_locked.py <task> <method>
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from ik_redistribution import summarize_ik_log
from sweep_grid_libero import JOINTS, FAULT_CONDITIONS, N_TEST, TEST_START_SEED, TASKS

task_name, mode = sys.argv[1], sys.argv[2]
assert task_name in TASKS, f"unknown task {task_name!r}"
assert mode in ("b1", "eci", "select", "random_n", "ik", "ik_pose"), mode
OUT_DIR = f"/workspace/results/libero_fault_sweep_locked_{mode}"
os.makedirs(OUT_DIR, exist_ok=True)

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=f"/workspace/results/_sweep_scratch_locked_{mode}_{task_name}")
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

for joint_name in JOINTS:
    for fault_type, severity in FAULT_CONDITIONS:
        if fault_type != "locked":
            continue
        fname = f"{task_name}_{joint_name}_{fault_type}_{severity}.json".replace("None", "na")
        out_path = os.path.join(OUT_DIR, fname)
        if os.path.exists(out_path):
            print(f"\nSKIP (exists): {mode}/{fname}", flush=True)
            continue
        runner = FaultRobomimicImageRunner(
            output_dir=f"/workspace/results/_sweep_run_locked_{mode}_{task_name}",
            dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
            n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key="agentview_image", abs_action=True,
            actuation_mode="joint", joint_kp=TASKS[task_name].get("joint_kp", 150),
        )
        if mode == "b1":
            policy = NativeJointPolicy(base_policy, base_seed=42, fault_spec=None)
        else:
            kw = dict(base_seed=42, mode=("ik" if mode == "ik_pose" else mode), env_ref=runner.env,
                      fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity)
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
                       "joint": joint_name, "fault_type": fault_type, "n_test": N_TEST,
                       "ik_stats": summarize_ik_log(policy)}, f, indent=2)
        print(f"\nRESULT {task_name}/{joint_name}/locked/{mode}: {score}", flush=True)
print(f"\nDONE {task_name} locked {mode}", flush=True)
