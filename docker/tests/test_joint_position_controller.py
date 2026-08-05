"""
Tests whether bypassing OSC_POSE's task-space Jacobian control (which is
unaware of the physical lock) with a per-joint JOINT_POSITION controller
eliminates the instability observed under joint3-locked. We command our
IK's own q_sol directly as joint targets (not an EE-pose that some other
controller must re-solve).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import robosuite
from robosuite.controllers import load_controller_config
from panda_kinematics import panda_fk
from ik_projector import solve_batch_ik
import torch

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

ctrl_cfg = load_controller_config(default_controller="JOINT_POSITION")
print("controller config:", ctrl_cfg)

env = robosuite.make(
    "Lift", robots="Panda", has_renderer=False,
    has_offscreen_renderer=False, use_camera_obs=False,
    controller_configs=ctrl_cfg,
)
np.random.seed(0)
obs = env.reset()
sim = env.sim

joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
qpos_addrs = [sim.model.get_joint_qpos_addr(n) for n in joint_names]
dof_addrs = [sim.model.get_joint_qvel_addr(n) for n in joint_names]

q_current = torch.tensor([sim.data.qpos[a] for a in qpos_addrs], dtype=torch.float32)
q_lock = q_current[2].item()  # joint3
print("initial q:", q_current.numpy(), " q_lock (joint3):", q_lock)

# pick a real target: current EE pose shifted +5cm in x (same synthetic
# target as the earlier OSC test, for direct comparability)
pos0, rot0 = panda_fk(q_current.unsqueeze(0))
pos0 = pos0.squeeze(0)
rot0 = rot0.squeeze(0)
target_pos = pos0 + torch.tensor([0.05, 0.0, 0.0])

locked_mask = torch.zeros(7, dtype=torch.bool)
locked_mask[2] = True
q_lock_vec = torch.zeros(1, 7, dtype=torch.float32)
q_lock_vec[0, 2] = q_lock

K = 64
seeds = torch.rand(K, 7, dtype=torch.float32) * (q_hi - q_lo) + q_lo
seeds[:, 2] = q_lock
seeds[0] = q_current.clone(); seeds[0, 2] = q_lock
target_pos_b = target_pos.unsqueeze(0).expand(K, -1)
target_rot_b = rot0.unsqueeze(0).expand(K, -1, -1)
q_lock_vec_k = q_lock_vec.expand(K, -1)
q_ref_k = q_current.unsqueeze(0).expand(K, -1)
q_sol, pos_err, rot_err = solve_batch_ik(target_pos_b, target_rot_b, seeds, q_lo, q_hi, locked_mask, q_lock_vec_k, q_ref_k, iters=20, lambda_reg=0.05)
best = pos_err.argmin()
q_target = q_sol[best]
print("IK target joint config:", q_target.numpy(), " achieved pos_err (offline):", pos_err[best].item())

print(f"\n{'step':>4} {'ach_x':>9} {'ach_y':>9} {'ach_z':>9} {'gap_to_qtarget_FK(mm)':>22}")
target_fk_pos, _ = panda_fk(q_target.unsqueeze(0))
target_fk_pos = target_fk_pos.squeeze(0).numpy()

OUTPUT_MAX = 0.05  # rad/step, matches controller config's output_max
for step in range(60):
    # JOINT_POSITION is DELTA-only in robosuite (no absolute mode exists for
    # it, unlike OSC_POSE/OSC_POSITION's control_delta switch). Recompute the
    # remaining distance to q_target from the CURRENT actual qpos every step
    # (closed-loop), clip to the controller's per-step range, normalize to
    # [-1,1] as the action format requires.
    q_now = torch.tensor([sim.data.qpos[a] for a in qpos_addrs], dtype=torch.float32)
    delta = torch.clamp(q_target - q_now, -OUTPUT_MAX, OUTPUT_MAX)
    action_joints = (delta / OUTPUT_MAX).numpy()
    action = np.concatenate([action_joints, [-1.0]]).astype(np.float32)
    obs, reward, done, info = env.step(action)
    # enforce physical lock (same as FaultInjector.step)
    sim.data.qpos[qpos_addrs[2]] = q_lock
    sim.data.qvel[dof_addrs[2]] = 0.0
    sim.forward()

    ach = obs['robot0_eef_pos'] if 'robot0_eef_pos' in obs else None
    if ach is None:
        # raw robosuite obs key may differ; query via sim directly
        site_id = sim.model.site_name2id("gripper0_grip_site")
        ach = sim.data.site_xpos[site_id].copy()
    gap = np.linalg.norm(ach - target_fk_pos) * 1000
    print(f"{step:>4} {ach[0]:>9.4f} {ach[1]:>9.4f} {ach[2]:>9.4f} {gap:>22.2f}")

env.close()
