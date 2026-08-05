"""
Paired comparison: B1 (no projector) vs FAVOR (blend_floor=0.15), SAME
test_start_seed=10000 (matches Phase 3's sweep_grid.py), same n_test, same
joint/fault condition. This is the first comparison in this project that is
actually valid for a McNemar-style paired test later -- everything before
this used different seeds and/or different n between B1 and FAVOR runs.
"""
import sys, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from ik_projector import ProjectWaypoints
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
TEST_START_SEED = 10000
N_TEST = 20

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_paired")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

def make_runner(output_dir, joint_name):
    return FaultRobomimicImageRunner(
        output_dir=output_dir, dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name, fault_type="locked", fault_severity=None,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=N_TEST,
        max_steps=cfg.task.env_runner.max_steps,
        n_obs_steps=cfg.task.env_runner.n_obs_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        render_obs_key=cfg.task.env_runner.render_obs_key,
        abs_action=cfg.task.env_runner.abs_action,
    )

results = {}
for joint_name, joint_idx in [("robot0_joint1", 0), ("robot0_joint4", 3)]:
    # B1: same runner, plain base_policy (no projector at all)
    runner_b1 = make_runner(f"/workspace/results/_paired_{joint_name}_b1", joint_name)
    log_b1 = runner_b1.run(base_policy)
    score_b1 = log_b1.get("test/mean_score")

    # FAVOR: same seeds, blend_floor=0.15
    fault_spec = {'joint_idx': joint_idx, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}
    projector = ProjectWaypoints(K=64, iters=5)
    runner_favor = make_runner(f"/workspace/results/_paired_{joint_name}_favor", joint_name)
    favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, projector=projector,
                                    env_ref=runner_favor.env, blend_floor=0.15)
    log_favor = runner_favor.run(favor)
    score_favor = log_favor.get("test/mean_score")

    print(f"{joint_name}: B1={score_b1}  FAVOR(floor=0.15)={score_favor}  (n={N_TEST}, seed={TEST_START_SEED}, paired)")
    results[joint_name] = {"b1": score_b1, "favor": score_favor}

with open("/workspace/results/paired_comparison_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
