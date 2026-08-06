"""
Paired B1 vs FAVOR comparison in actuation_mode='joint'. Both paths call the
SAME project_waypoints_to_joint_targets function (no FK round-trip, raw q_sol
sent directly to JointActuationWrapper) -- B1 = fault_spec=None (unconstrained
terminal IK), FAVOR = fault_spec set (one joint held fixed in the IK solve).
Same seed for both -> valid paired comparison.
"""
import sys, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
TEST_START_SEED = 10000
N_TEST = 20

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_joint_paired")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

def make_runner(output_dir, joint_name, fault_type):
    return FaultRobomimicImageRunner(
        output_dir=output_dir, dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name, fault_type=fault_type, fault_severity=None,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=N_TEST,
        max_steps=cfg.task.env_runner.max_steps,
        n_obs_steps=cfg.task.env_runner.n_obs_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        render_obs_key=cfg.task.env_runner.render_obs_key,
        abs_action=cfg.task.env_runner.abs_action,
        actuation_mode='joint',
    )

results = {}
for joint_name, joint_idx in [("robot0_joint3", 2)]:
    # B1: same actuation_mode='joint', fault_spec=None -> unconstrained terminal IK
    runner_b1 = make_runner(f"/workspace/results/_jointpaired_{joint_name}_b1", joint_name, None)
    favor_b1 = FavorHybridImagePolicy(base_policy, fault_spec=None, env_ref=runner_b1.env,
                                       actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)
    log_b1 = runner_b1.run(favor_b1)
    score_b1 = log_b1.get("test/mean_score")

    # FAVOR: fault_spec set -> constrained terminal IK
    fault_spec = {'joint_idx': joint_idx, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}
    runner_favor = make_runner(f"/workspace/results/_jointpaired_{joint_name}_favor", joint_name, "locked")
    favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, env_ref=runner_favor.env,
                                    actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)
    log_favor = runner_favor.run(favor)
    score_favor = log_favor.get("test/mean_score")

    print(f"{joint_name} (JOINT mode): B1={score_b1}  FAVOR={score_favor}  (n={N_TEST}, seed={TEST_START_SEED}, paired)")
    results[joint_name] = {"b1": score_b1, "favor": score_favor}

with open("/workspace/results/joint_mode_paired_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
