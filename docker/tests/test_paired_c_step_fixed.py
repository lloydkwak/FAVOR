"""
Re-verification of the ORIGINAL C-step (ECI, mid-denoising projection) +
OSC execution, now that lambda_reg=0.0 fixes the IK convergence bug that
was silently present through all three prior C-step redesigns this
project has gone through (schedule floor, joint-space regularization,
actual-qpos seeding). Paired B1 vs FAVOR, joint3 locked, n=20.
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
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_cstep_paired")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

def make_runner(output_dir):
    return FaultRobomimicImageRunner(
        output_dir=output_dir, dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
        fault_joint_name="robot0_joint3", fault_type="locked", fault_severity=None,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=N_TEST,
        max_steps=cfg.task.env_runner.max_steps,
        n_obs_steps=cfg.task.env_runner.n_obs_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        render_obs_key=cfg.task.env_runner.render_obs_key,
        abs_action=cfg.task.env_runner.abs_action,
        actuation_mode='osc',
    )

# B1: OSC, no correction at all
runner_b1 = make_runner("/workspace/results/_cstep_paired_b1")
favor_b1 = FavorHybridImagePolicy(base_policy, fault_spec=None, env_ref=runner_b1.env)
log_b1 = runner_b1.run(favor_b1)
score_b1 = log_b1.get("test/mean_score")
print(f"B1 (OSC, no correction): {score_b1}")

# FAVOR: original C-step (ECI), now with lambda_reg=0.0, + OSC execution
fault_spec = {'joint_idx': 2, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}
runner_favor = make_runner("/workspace/results/_cstep_paired_favor")
projector = ProjectWaypoints(K=64, iters=5, lambda_reg=0.0)
favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, projector=projector, env_ref=runner_favor.env)
log_favor = runner_favor.run(favor)
score_favor = log_favor.get("test/mean_score")
print(f"FAVOR (original C-step/ECI, lambda_reg=0.0, OSC execution): {score_favor}")

results = {"b1_osc": score_b1, "favor_cstep_fixed": score_favor, "n": N_TEST}
with open("/workspace/results/cstep_paired_fixed_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print("\n(historical result before this session's lambda_reg fix: b1=0.39, favor=0.36, n=28)")
