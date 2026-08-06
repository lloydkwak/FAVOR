"""
Distinguishes two hypotheses for the persistent z-oscillation in joint mode:
(A) execution-layer issue (IK/controller still not converging well), or
(B) the POLICY ITSELF is producing inconsistent/oscillating raw EE-pose
    predictions, because it was trained on OSC-actuated trajectories and
    the actual observed state sequence under JOINT_POSITION control looks
    out-of-distribution (stiffer motion, different velocity profile).

Logs, per predict_action call: the policy's RAW predicted EE-pose (BEFORE
any IK conversion) for the first and last waypoint, plus the achieved eef_pos
after execution. If (B), raw_pred should itself jump around a lot between
consecutive calls, not just the executed trajectory.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_consistency")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_consistency_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type=None, fault_severity=None,
    n_train=0, n_test=1, test_start_seed=10000, n_envs=1,
    max_steps=200,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
    actuation_mode='joint',
)
favor = FavorHybridImagePolicy(base_policy, fault_spec=None, env_ref=runner.env,
                                actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)

# Capture the RAW EE-pose prediction (pre-IK) by hooking action_pred, which
# favor_policy.py already returns alongside the joint-target action.
raw_pred_trace = []
z_trace = []
gripper_trace = []

obs = runner.env.reset()
favor.reset()
device = base_policy.device
done = False
step_count = 0
call_idx = 0
while not done and step_count < 200:
    obs_dict = {k: torch.from_numpy(v).to(device=device) for k, v in dict(obs).items()}
    with torch.no_grad():
        action_dict = favor.predict_action(obs_dict)
    action_pred = action_dict['action_pred'][0].detach().cpu().numpy()  # (16, 10) raw EE-pose pred, ALL waypoints
    raw_pred_trace.append(action_pred[-1, 0:3].copy())  # last waypoint's predicted xyz
    gripper_trace.append(action_dict['action'][0, :, -1].detach().cpu().numpy().copy())

    action = action_dict['action'].detach().to('cpu').numpy()
    obs, reward, done, info = runner.env.step(action)
    z_trace.append(obs['robot0_eef_pos'][0, -1, 2])
    done = np.all(done)
    step_count += action.shape[1]
    call_idx += 1

raw_pred_trace = np.array(raw_pred_trace)
z_trace = np.array(z_trace)

print("call-to-call jump in policy's RAW predicted xyz (last waypoint), mm:")
jumps = np.linalg.norm(np.diff(raw_pred_trace, axis=0), axis=1) * 1000
print(jumps.round(1))
print(f"\nmean jump: {jumps.mean():.1f}mm  max jump: {jumps.max():.1f}mm  std: {jumps.std():.1f}mm")

print("\nraw predicted z (last waypoint) per call:")
print(raw_pred_trace[:, 2].round(3))

print("\nactual achieved z per macro-step:")
print(z_trace.round(3))

print("\ngripper command, last value of each call:")
print([g[-1].round(3) for g in gripper_trace])
print("min gripper value ever commanded:", min(g.min() for g in gripper_trace))
