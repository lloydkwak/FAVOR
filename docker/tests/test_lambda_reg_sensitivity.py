import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch
from panda_kinematics import panda_fk
from ik_projector import solve_batch_ik

torch.manual_seed(0)
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

# realistic scenario: q_ref = actual current pose, target = a plausible nearby waypoint
q_ref_val = torch.tensor([-0.0185, 0.1823, -0.0428, -2.6306, 0.0120, 2.9928, 0.7933], dtype=torch.float32)
q_target_true = q_ref_val.clone()
q_target_true[3] = -2.5  # joint4 moves slightly (simulating a real small-motion waypoint)
target_pos, target_rot = panda_fk(q_target_true.unsqueeze(0))
target_pos, target_rot = target_pos.squeeze(0), target_rot.squeeze(0)

K = 64
locked_mask = torch.zeros(7, dtype=torch.bool)
locked_mask[3] = True
q_lock_vec = torch.zeros(K, 7, dtype=torch.float32)
q_lock_vec[:, 3] = -2.630583377209979  # locked value, close to q_ref's own joint4

target_pos_b = target_pos.unsqueeze(0).expand(K, -1)
target_rot_b = target_rot.unsqueeze(0).expand(K, -1, -1)
q_ref_b = q_ref_val.unsqueeze(0).expand(K, -1).clone()

for iters in [5, 10, 20, 50]:
    for lam in [0.0, 0.05, 0.1, 0.5]:
        seeds = torch.rand(K, 7, dtype=torch.float32) * (q_hi - q_lo) + q_lo
        seeds[:, 3] = -2.630583377209979
        seeds[0] = q_ref_val.clone(); seeds[0,3] = -2.630583377209979
        q_sol, pos_err, rot_err = solve_batch_ik(target_pos_b, target_rot_b, seeds, q_lo, q_hi,
                                                   locked_mask, q_lock_vec, q_ref_b, iters=iters, lambda_reg=lam)
        best = pos_err.argmin()
        print(f"iters={iters:3d}  lambda={lam:.2f}  best_pos_err={pos_err[best].item():.4f}  best_rot_err={rot_err[best].item():.4f}")
