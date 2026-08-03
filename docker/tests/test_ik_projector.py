import sys
sys.path.insert(0, "/workspace/docker")
import torch
from panda_kinematics import panda_fk
from ik_projector import solve_batch_ik

torch.manual_seed(0)
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float64)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float64)

# --- Test 1: no fault, target from a known-reachable q, IK should converge ---
q_true = torch.tensor([0.1, 0.3, 0.0, -2.0, 0.0, 2.0, 0.7], dtype=torch.float64)
target_pos, target_rot = panda_fk(q_true.unsqueeze(0))
target_pos, target_rot = target_pos.squeeze(0), target_rot.squeeze(0)

K = 64
no_lock_mask = torch.zeros(7, dtype=torch.bool)
no_lock_vec = torch.zeros(7, dtype=torch.float64)

seeds = torch.rand(K, 7, dtype=torch.float64) * (q_hi - q_lo) + q_lo
q_sol, pos_err = solve_batch_ik(target_pos, target_rot, seeds, q_lo, q_hi, no_lock_mask, no_lock_vec, iters=20)
best = pos_err.argmin()
print("Test1 (no fault) best pos_err:", pos_err[best].item())
t1_pass = pos_err[best].item() < 0.01

# --- Test 2: joint3 locked at q_true[2] (reachable), lock value should hold exactly ---
locked_mask2 = torch.zeros(7, dtype=torch.bool)
locked_mask2[2] = True
q_lock_val = q_true[2].item()
q_lock_vec2 = torch.zeros(7, dtype=torch.float64)
q_lock_vec2[2] = q_lock_val

seeds2 = torch.rand(K, 7, dtype=torch.float64) * (q_hi - q_lo) + q_lo
seeds2[:, 2] = q_lock_val
q_sol2, pos_err2 = solve_batch_ik(target_pos, target_rot, seeds2, q_lo, q_hi, locked_mask2, q_lock_vec2, iters=20)
best2 = pos_err2.argmin()
print("Test2 (joint3 locked at", q_lock_val, ", reachable) best pos_err:", pos_err2[best2].item())
print("  locked joint values (should all equal", q_lock_val, "):", q_sol2[:, 2].unique())
t2_pass = pos_err2[best2].item() < 0.01
t2_locked_held = torch.allclose(q_sol2[:, 2], torch.full((K,), q_lock_val, dtype=torch.float64), atol=1e-9)

# --- Test 3: joint3 locked at a DIFFERENT value -> target now unreachable exactly,
#     lock must still hold even though pos_err stays > 0 ---
q_lock_val3 = q_lock_val + 0.5
q_lock_vec3 = torch.zeros(7, dtype=torch.float64)
q_lock_vec3[2] = q_lock_val3

seeds3 = torch.rand(K, 7, dtype=torch.float64) * (q_hi - q_lo) + q_lo
seeds3[:, 2] = q_lock_val3
q_sol3, pos_err3 = solve_batch_ik(target_pos, target_rot, seeds3, q_lo, q_hi, locked_mask2, q_lock_vec3, iters=20)
t3_locked_held = torch.allclose(q_sol3[:, 2], torch.full((K,), q_lock_val3, dtype=torch.float64), atol=1e-9)
print("Test3 (joint3 locked at", q_lock_val3, ", DIFFERENT/unreachable) min pos_err:", pos_err3.min().item(),
      " (expected > 0)")
print("  locked joint held at", q_lock_val3, ":", t3_locked_held)

print("=" * 60)
print("t1_pass", t1_pass, " t2_pass", t2_pass, " t2_locked_held", t2_locked_held, " t3_locked_held", t3_locked_held)
if t1_pass and t2_pass and t2_locked_held and t3_locked_held:
    print("IK_PROJECTOR_VERIFIED")
else:
    sys.exit(1)
