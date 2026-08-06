"""
Directly tests the "IK solver itself is broken" hypothesis using REAL policy
output during an actual B1 (no-fault) rollout -- not synthetic data. For
each predict_action call: computes FK(q_targets) and compares against the
policy's own raw EE-pose targets (action[...,0:9]) to get per-waypoint
position/rotation residuals. If residuals are small (~cm-level), the IK
solver is NOT the problem. If large, it is.

Also logs actual eef trajectory and gripper to see whether the per-waypoint
restoration (vs the earlier final-waypoint-only bug) changed the qualitative
behavior.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from favor_fault_runner import FaultRobomimicImageRunner
from panda_kinematics import panda_fk
from embodiment_guidance import rotmat_to_axis_angle_stable
from ik_projector import _rot6d_to_matrix

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_ikconv")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_ikconv_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type=None, fault_severity=None,
    n_train=0, n_test=1, test_start_seed=10000, n_envs=1,
    max_steps=160,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
    actuation_mode='joint',
)
favor = FavorHybridImagePolicy(base_policy, fault_spec=None, env_ref=runner.env,
                                actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)

obs = runner.env.reset()
favor.reset()
device = base_policy.device
done = False
step_count = 0
call_idx = 0
z_trace = []
gripper_trace = []

while not done and step_count < 160:
    obs_dict = {k: torch.from_numpy(v).to(device=device) for k, v in dict(obs).items()}
    with torch.no_grad():
        action_dict = favor.predict_action(obs_dict)
    action = action_dict['action'].detach().to('cpu')          # (1, 8, 8) q_targets+gripper
    action_pred = action_dict['action_pred'][0].detach().cpu() # (16, 10) raw EE-pose pred, all waypoints
    n_steps = action.shape[1]

    # Independent IK-convergence check: FK(q_targets) vs the policy's own
    # intended EE-pose for those same waypoints (first n_steps of action_pred).
    q_targets = action[0, :, :7]  # (n_steps, 7)
    fk_pos, fk_rot = panda_fk(q_targets)
    intended_pos = action_pred[:n_steps, 0:3]
    intended_rot = _rot6d_to_matrix(action_pred[:n_steps, 3:9])
    pos_err = (fk_pos - intended_pos).norm(dim=-1) * 1000  # mm
    rot_err_mat = torch.matmul(intended_rot, fk_rot.transpose(-1, -2))
    rot_err = rotmat_to_axis_angle_stable(rot_err_mat).norm(dim=-1) * 180 / np.pi  # deg

    print(f"call {call_idx}: pos_err(mm) per waypoint: {[f'{v:.1f}' for v in pos_err.tolist()]}")
    print(f"          rot_err(deg) per waypoint: {[f'{v:.1f}' for v in rot_err.tolist()]}")

    action_np = action.numpy()
    obs, reward, done, info = runner.env.step(action_np)
    z_trace.append(obs['robot0_eef_pos'][0, -1, 2])
    gripper_trace.append(action_np[0, -1, -1])
    done = np.all(done)
    step_count += n_steps
    call_idx += 1

print()
print("z trajectory:", [f"{v:.3f}" for v in z_trace])
print("gripper command (last of each call):", [f"{v:.2f}" for v in gripper_trace])
