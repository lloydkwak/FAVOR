import sys
sys.path.insert(0, "/workspace/docker")
import torch
from panda_kinematics import panda_fk, panda_jacobian

torch.manual_seed(1)
q0 = torch.tensor([0.2, 0.3, -0.1, -2.0, 0.15, 2.1, 0.6], dtype=torch.float64)

J_closed, pos0, rot0 = panda_jacobian(q0.unsqueeze(0))
J_closed = J_closed.squeeze(0)  # (6,7)
print("J_closed shape:", J_closed.shape)

eps = 1e-6
J_num_pos = torch.zeros(3, 7, dtype=torch.float64)
J_num_rot = torch.zeros(3, 7, dtype=torch.float64)
for i in range(7):
    dq = torch.zeros(7, dtype=torch.float64)
    dq[i] = eps
    pos_p, rot_p = panda_fk((q0 + dq).unsqueeze(0))
    pos_m, rot_m = panda_fk((q0 - dq).unsqueeze(0))
    J_num_pos[:, i] = (pos_p.squeeze(0) - pos_m.squeeze(0)) / (2 * eps)

    # numerical angular velocity column: d(rot)/dtheta_i expressed as skew(w) = dR * R^T / dtheta
    dR = (rot_p.squeeze(0) - rot_m.squeeze(0)) / (2 * eps)
    W = dR @ rot0.squeeze(0).transpose(-1, -2)
    w = torch.stack([W[2,1], W[0,2], W[1,0]])
    J_num_rot[:, i] = w

print("=== Position rows: closed-form vs numerical, per-joint abs max diff ===")
diff_pos = (J_closed[0:3] - J_num_pos).abs()
for i in range(7):
    print(f"  joint{i+1}: max_diff={diff_pos[:,i].max().item():.6f}")

print("=== Rotation rows: closed-form vs numerical, per-joint abs max diff ===")
diff_rot = (J_closed[3:6] - J_num_rot).abs()
for i in range(7):
    print(f"  joint{i+1}: max_diff={diff_rot[:,i].max().item():.6f}")

print("overall max diff pos:", diff_pos.max().item(), " rot:", diff_rot.max().item())
