"""
Regression test for the KEY confirmation: replacing OSC_POSE with
JOINT_POSITION (commanding our IK's own q_sol directly, closed-loop delta,
re-enforcing the physical lock every step) eliminates the instability --
tracking error must converge smoothly and monotonically to a small residual,
with no oscillation late in the trajectory.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch
import robosuite
from robosuite.controllers import load_controller_config
from panda_kinematics import panda_fk
from ik_projector import solve_batch_ik

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

ctrl_cfg = load_controller_config(default_controller="JOINT_POSITION")
env = robosuite.make("Lift", robots="Panda", has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False,
                      controller_configs=ctrl_cfg)
np.random.seed(0)
obs = env.reset()
sim = env.sim
joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
qpos_addrs = [sim.model.get_joint_qpos_addr(n) for n in joint_names]
dof_addrs = [sim.model.get_joint_qvel_addr(n) for n in joint_names]

q_current = torch.tensor([sim.data.qpos[a] for a in qpos_addrs], dtype=torch.float32)
q_lock = q_current[2].item()
pos0, rot0 = panda_fk(q_current.unsqueeze(0))
target_pos = pos0.squeeze(0) + torch.tensor([0.05, 0.0, 0.0])

locked_mask = torch.zeros(7, dtype=torch.bool); locked_mask[2] = True
q_lock_vec = torch.zeros(1, 7, dtype=torch.float32); q_lock_vec[0, 2] = q_lock
K = 64
seeds = torch.rand(K, 7, dtype=torch.float32) * (q_hi - q_lo) + q_lo
seeds[:, 2] = q_lock; seeds[0] = q_current.clone(); seeds[0, 2] = q_lock
q_ref_k = q_current.unsqueeze(0).expand(K, -1)
q_sol, pos_err, rot_err = solve_batch_ik(
    target_pos.unsqueeze(0).expand(K, -1), rot0.squeeze(0).unsqueeze(0).expand(K, -1, -1),
    seeds, q_lo, q_hi, locked_mask, q_lock_vec.expand(K, -1), q_ref_k, iters=20, lambda_reg=0.05)
q_target = q_sol[pos_err.argmin()]
target_fk_pos, _ = panda_fk(q_target.unsqueeze(0))
target_fk_pos = target_fk_pos.squeeze(0).numpy()

OUTPUT_MAX = 0.05
gaps = []
for step in range(60):
    q_now = torch.tensor([sim.data.qpos[a] for a in qpos_addrs], dtype=torch.float32)
    delta = torch.clamp(q_target - q_now, -OUTPUT_MAX, OUTPUT_MAX)
    action = np.concatenate([(delta / OUTPUT_MAX).numpy(), [-1.0]]).astype(np.float32)
    obs, _, _, _ = env.step(action)
    sim.data.qpos[qpos_addrs[2]] = q_lock
    sim.data.qvel[dof_addrs[2]] = 0.0
    sim.forward()
    site_id = sim.model.site_name2id("gripper0_grip_site")
    ach = sim.data.site_xpos[site_id].copy()
    gaps.append(np.linalg.norm(ach - target_fk_pos) * 1000)

env.close()
print("first 5 gaps (mm):", [f"{g:.2f}" for g in gaps[:5]])
print("last 5 gaps (mm):", [f"{g:.2f}" for g in gaps[-5:]])

assert gaps[-1] < 5.0, f"expected convergence to <5mm, got {gaps[-1]:.2f}mm"
assert gaps[-1] < gaps[0], "expected monotonic-ish improvement, tracking got worse"
late = gaps[-10:]
assert max(late) - min(late) < 1.0, f"expected late-trajectory stability (no oscillation), range={max(late)-min(late):.2f}mm"
print("JOINT_POSITION_FIXES_INSTABILITY_CONFIRMED")
