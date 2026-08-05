"""
Realistic (non-pathological) tracking-gap test: runs an ACTUAL FAVOR rollout
(joint3 locked, real diffusion policy predict_action loop, continuously
updated targets exactly as in real usage) and logs, at each low-level physics
step, the COMMANDED EE-pose (what our corrected action says) vs the ACHIEVED
EE-pose (what the robot actually reaches), plus the achieved joint config vs
our IK's chosen q_sol. This directly tests whether "IK-feasible correction"
actually reaches the robot, under realistic conditions (not the repeated
static-target pathology from the earlier failed diagnostic).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from ik_projector import ProjectWaypoints
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_realistic_gap")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
import os
USE_FAULT = os.environ.get("USE_FAULT", "1") == "1"
fault_type_arg = "locked" if USE_FAULT else None
fault_spec = {'joint_idx': 2, 'q_lock': 0.0, 'fault_type': 'locked' if USE_FAULT else None, 'q_lo': q_lo, 'q_hi': q_hi} if USE_FAULT else None

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_realistic_gap",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type=fault_type_arg, fault_severity=None,
    n_train=0, n_test=1, test_start_seed=10000, n_envs=1,
    max_steps=80,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
)
print("USE_FAULT =", USE_FAULT)
if USE_FAULT:
    projector = ProjectWaypoints(K=64, iters=5, lambda_reg=0.5)
    favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, projector=projector,
                                    env_ref=runner.env, blend_floor=0.15)
else:
    favor = base_policy  # plain B1, no fault, no projector -- pure baseline tracking behavior

# monkeypatch AsyncVectorEnv.step to log commanded vs achieved before/after
orig_step = runner.env.step
gaps = []
def traced_step(action):
    # action shape: (n_envs, n_action_steps, action_dim) -- unnormalized abs
    # action from undo_transform_action, columns 0:3=pos, 3:9=rot6d... wait,
    # by the time it reaches env.step it's already converted to robosuite's
    # native [pos(3), axis_angle(3), gripper(1)] via undo_transform_action.
    commanded_pos = action[0, 0, 0:3].copy()
    result = orig_step(action)
    obs = result[0]
    achieved_pos = obs['robot0_eef_pos'][0, -1].copy()  # (n_envs, T, 3) -> last obs step
    gap = np.linalg.norm(achieved_pos - commanded_pos)
    gaps.append(gap)
    return result
runner.env.step = traced_step

log = runner.run(favor)
print("score:", log.get("test/mean_score"))
gaps = np.array(gaps)
print(f"tracking gap over {len(gaps)} steps: mean={gaps.mean()*1000:.2f}mm  "
      f"median={np.median(gaps)*1000:.2f}mm  max={gaps.max()*1000:.2f}mm  "
      f"final={gaps[-1]*1000:.2f}mm")
print("first 10 gaps (mm):", (gaps[:10]*1000).round(2))
print("last 10 gaps (mm):", (gaps[-10:]*1000).round(2))
