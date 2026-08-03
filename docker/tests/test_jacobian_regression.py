"""
Regression test for panda_jacobian(): closed-form Jacobian must match
finite-difference numerically, for ALL 7 joints (this exact check caught the
modified-DH joint-axis bug where only joint1 (alpha=0) was accidentally
correct). Run this after any future change to panda_kinematics.py.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import torch
from panda_kinematics import panda_fk, panda_jacobian

torch.manual_seed(1)
q0 = torch.tensor([0.2, 0.3, -0.1, -2.0, 0.15, 2.1, 0.6], dtype=torch.float64)

J_closed, pos0, rot0 = panda_jacobian(q0.unsqueeze(0))
J_closed = J_closed.squeeze(0)

eps = 1e-6
J_num_pos = torch.zeros(3, 7, dtype=torch.float64)
J_num_rot = torch.zeros(3, 7, dtype=torch.float64)
for i in range(7):
    dq = torch.zeros(7, dtype=torch.float64)
    dq[i] = eps
    pos_p, rot_p = panda_fk((q0 + dq).unsqueeze(0))
    pos_m, rot_m = panda_fk((q0 - dq).unsqueeze(0))
    J_num_pos[:, i] = (pos_p.squeeze(0) - pos_m.squeeze(0)) / (2 * eps)
    dR = (rot_p.squeeze(0) - rot_m.squeeze(0)) / (2 * eps)
    W = dR @ rot0.squeeze(0).transpose(-1, -2)
    J_num_rot[:, i] = torch.stack([W[2, 1], W[0, 2], W[1, 0]])

diff_pos = (J_closed[0:3] - J_num_pos).abs().max().item()
diff_rot = (J_closed[3:6] - J_num_rot).abs().max().item()
print("max diff pos:", diff_pos, " rot:", diff_rot)

TOL = 1e-4
if diff_pos > TOL or diff_rot > TOL:
    print("JACOBIAN_REGRESSION_FAILED")
    sys.exit(1)
print("JACOBIAN_REGRESSION_OK")
