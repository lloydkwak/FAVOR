"""
n_test=50 confirmation run for the 5 conditions where select improved over
B1 at n_test=20 (fault_sweep_libero_phase2_select_results.csv). Runs B1,
select, and random_n back-to-back for each condition, all at n_test=50
with the SAME TEST_START_SEED=10000 base (the first 20 of these 50 seeds
are the exact same episodes already run at n_test=20 -- FaultRobomimicImageRunner
generates seeds sequentially from test_start_seed, so this is a superset,
not a fresh independent sample).

Purpose: at n=20, only 2/5 conditions reached significance (Fisher's exact,
approximate) for select vs B1, and only for select vs random_n. Larger n
gives more statistical power to confirm the conditions that were
directionally correct but not yet significant (milk/joint7,
bowl_ramekin/joint7) and firms up the two that were borderline
(alphabet_soup/joint5's select-vs-random_n p=0.056).

Saves per-episode success/failure (test/sim_max_reward_{seed}) alongside
the aggregate score in every result file, enabling a real paired McNemar
test this time (not the independent-samples Fisher's exact approximation
used for the n=20 results).

Usage: python run_libero_confirm_n50.py
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid_libero import TASKS, TEST_START_SEED

CONDITIONS = [
    ("alphabet_soup", "robot0_joint5"),
    ("milk", "robot0_joint5"),
    ("milk", "robot0_joint7"),
    ("bowl_ramekin", "robot0_joint6"),
    ("bowl_ramekin", "robot0_joint7"),
]
N_TEST = 50
OUT_DIR = "/workspace/results/libero_confirm_n50"
os.makedirs(OUT_DIR, exist_ok=True)

_policy_cache = {}
def get_base_policy(task_name):
    if task_name not in _policy_cache:
        payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        ws = cls(cfg, output_dir=f"/workspace/results/_confirm_scratch_{task_name}")
        ws.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = ws.ema_model if cfg.training.use_ema else ws.model
        policy.to(torch.device("cuda:0"))
        policy.eval()
        _policy_cache[task_name] = (policy, cfg)
    return _policy_cache[task_name]


def run_one(task_name, joint_name, mode):
    fname = f"{task_name}_{joint_name}_locked_{mode}_n50.json"
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"SKIP (exists): {fname}")
        return

    base_policy, cfg = get_base_policy(task_name)
    runner = FaultRobomimicImageRunner(
        output_dir=f"/workspace/results/_confirm_run_{task_name}_{mode}",
        dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name, fault_type="locked", fault_severity=None,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
        max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        render_obs_key="agentview_image", abs_action=True,
        actuation_mode="joint", joint_kp=150,
    )
    if mode == "b1":
        policy = NativeJointPolicy(base_policy, base_seed=42, fault_spec=None)
    elif mode == "select":
        policy = NativeJointPolicy(
            base_policy, base_seed=42, mode='select', env_ref=runner.env,
            fault_joint_name=joint_name, fault_type="locked", fault_severity=None,
            n_select=32, chunk_size=8,
        )
    elif mode == "random_n":
        policy = NativeJointPolicy(
            base_policy, base_seed=42, mode='random_n', env_ref=runner.env,
            fault_joint_name=joint_name, fault_type="locked", fault_severity=None,
            n_select=32, chunk_size=8,
        )
    else:
        raise ValueError(mode)

    log = runner.run(policy)
    score = log.get("test/mean_score")
    per_episode = {k.replace("test/sim_max_reward_", ""): v
                   for k, v in log.items() if k.startswith("test/sim_max_reward_")}
    print(f"  {task_name}/{joint_name}/locked/{mode} (n=50): {score}")

    with open(out_path, "w") as f:
        json.dump({mode: score, "per_episode": per_episode}, f, indent=2)
    print(f"SAVED: {fname}")


for task_name, joint_name in CONDITIONS:
    for mode in ("b1", "select", "random_n"):
        run_one(task_name, joint_name, mode)

print("CONFIRM N=50 SWEEP COMPLETE")
