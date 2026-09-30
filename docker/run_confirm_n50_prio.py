"""
Fresh-seed replication (n=50, seeds 10020-10069) for the Priority-IK main claim.
Usage: python run_confirm_n50_prio.py <task>
Runs, for every selected locked condition of <task>: b1, ik (B-IK pos), ik_pose (B-IK pose), prio.
Output: results/libero_confirm_n50_fresh/<task>_<joint>_locked_<method>_n50.json  (SKIP if exists)
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import favor_fault_runner  # noqa: F401
from favor_fault_runner import FaultRobomimicImageRunner
from native_joint_policy import NativeJointPolicy
from native_joint_policy_prio import PrioIKPolicy
from sweep_grid_libero import TASKS

SEL = {"alphabet_soup": [1], "milk": [3, 6, 7], "bowl_stove": [1, 7], "bowl_ramekin": [6]}
METHODS = ["b1", "ik", "ik_pose", "prio"]
N_TEST, START = 50, 10020
task_name = sys.argv[1]
assert task_name in SEL, task_name
OUT_DIR = "/workspace/results/libero_confirm_n50_fresh"
os.makedirs(OUT_DIR, exist_ok=True)

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=f"/workspace/results/_confirm_scratch_{task_name}")
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to(torch.device("cuda:0")); base_policy.eval()

for j in SEL[task_name]:
    joint_name = f"robot0_joint{j}"
    for m in METHODS:
        out_path = os.path.join(OUT_DIR, f"{task_name}_{joint_name}_locked_{m}_n50.json")
        if os.path.exists(out_path):
            print(f"\nSKIP (exists): {os.path.basename(out_path)}", flush=True); continue
        runner = FaultRobomimicImageRunner(
            output_dir=f"/workspace/results/_confirm_run_{task_name}",
            dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
            fault_joint_name=joint_name, fault_type="locked", fault_severity=None,
            n_train=0, n_test=N_TEST, test_start_seed=START, n_envs=5,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key="agentview_image", abs_action=True,
            actuation_mode="joint", joint_kp=TASKS[task_name].get("joint_kp", 150),
        )
        kw = dict(base_seed=42, env_ref=runner.env, fault_joint_name=joint_name,
                  fault_type="locked", fault_severity=None)
        if m == "b1":
            policy = NativeJointPolicy(base_policy, base_seed=42, fault_spec=None)
        elif m == "ik":
            policy = NativeJointPolicy(base_policy, mode="ik", **kw)
        elif m == "ik_pose":
            policy = NativeJointPolicy(base_policy, mode="ik", ik_overrides={"rot_weight": 1.0}, **kw)
        else:
            policy = PrioIKPolicy(base_policy, lam2=0.2, **kw)
        log = runner.run(policy)
        score = log.get("test/mean_score")
        per_episode = {k.replace("test/sim_max_reward_", ""): v
                       for k, v in log.items() if k.startswith("test/sim_max_reward_")}
        with open(out_path, "w") as f:
            json.dump({m: score, "per_episode": per_episode, "task": task_name, "joint": joint_name,
                       "fault_type": "locked", "n_test": N_TEST, "test_start_seed": START}, f, indent=2)
        print(f"\nRESULT {task_name}/{joint_name}/locked/{m}/n50: {score}", flush=True)
print(f"\nDONE {task_name} n50", flush=True)
