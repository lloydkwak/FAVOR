import sys
sys.path.insert(0, "/workspace/docker")
import torch
from panda_kinematics import panda_fk
from ik_projector import solve_batch_ik

torch.manual_seed(0)
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

q_true = torch.tensor([0.1, 0.3, 0.0, -2.0, 0.0, 2.0, 0.7], dtype=torch.float32)
target_pos, target_rot = panda_fk(q_true.unsqueeze(0))
target_pos, target_rot = target_pos.squeeze(0), target_rot.squeeze(0)

K = 64
no_lock_mask = torch.zeros(7, dtype=torch.bool)
no_lock_vec = torch.zeros(K, 7, dtype=torch.float32)
q_ref_zero = torch.zeros(K, 7, dtype=torch.float32)  # unused when lambda_reg=0.0 -- tests 1-3 isolate pure convergence/lock-holding, separate from Test4's regularization-pull check

target_pos_b = target_pos.unsqueeze(0).expand(K, -1)
target_rot_b = target_rot.unsqueeze(0).expand(K, -1, -1)

# --- Test 1: no fault, no meaningful q_ref pull (lambda small relative to problem) ---
seeds = torch.rand(K, 7, dtype=torch.float32) * (q_hi - q_lo) + q_lo
q_sol, pos_err, rot_err = solve_batch_ik(target_pos_b, target_rot_b, seeds, q_lo, q_hi,
                                          no_lock_mask, no_lock_vec, q_ref_zero, iters=30, lambda_reg=0.0)
best = pos_err.argmin()
print("Test1 (no fault) best pos_err:", pos_err[best].item(), " rot_err:", rot_err[best].item())
t1_pass = pos_err[best].item() < 0.01 and rot_err[best].item() < 0.15

# --- Test 2: joint3 locked, reachable ---
locked_mask2 = torch.zeros(7, dtype=torch.bool)
locked_mask2[2] = True
q_lock_val = q_true[2].item()
q_lock_vec2 = torch.zeros(K, 7, dtype=torch.float32)
q_lock_vec2[:, 2] = q_lock_val
q_ref2 = torch.zeros(K, 7, dtype=torch.float32)

seeds2 = torch.rand(K, 7, dtype=torch.float32) * (q_hi - q_lo) + q_lo
seeds2[:, 2] = q_lock_val
q_sol2, pos_err2, rot_err2 = solve_batch_ik(target_pos_b, target_rot_b, seeds2, q_lo, q_hi,
                                             locked_mask2, q_lock_vec2, q_ref2, iters=30, lambda_reg=0.0)
best2 = pos_err2.argmin()
print("Test2 (joint3 locked, reachable) best pos_err:", pos_err2[best2].item(), " rot_err:", rot_err2[best2].item())
t2_pass = pos_err2[best2].item() < 0.01 and rot_err2[best2].item() < 0.15
t2_locked_held = torch.allclose(q_sol2[:, 2], torch.full((K,), q_lock_val, dtype=torch.float32), atol=1e-5)

# --- Test 3: different per-sample lock values ---
half = K // 2
q_lock_val_a, q_lock_val_b = 0.3, -0.9
q_lock_vec3 = torch.zeros(K, 7, dtype=torch.float32)
q_lock_vec3[:half, 2] = q_lock_val_a
q_lock_vec3[half:, 2] = q_lock_val_b
q_ref3 = torch.zeros(K, 7, dtype=torch.float32)
seeds3 = torch.rand(K, 7, dtype=torch.float32) * (q_hi - q_lo) + q_lo
seeds3[:half, 2] = q_lock_val_a
seeds3[half:, 2] = q_lock_val_b
q_sol3, pos_err3, rot_err3 = solve_batch_ik(target_pos_b, target_rot_b, seeds3, q_lo, q_hi,
                                             locked_mask2, q_lock_vec3, q_ref3, iters=30, lambda_reg=0.0)
held_a = torch.allclose(q_sol3[:half, 2], torch.full((half,), q_lock_val_a, dtype=torch.float32), atol=1e-5)
held_b = torch.allclose(q_sol3[half:, 2], torch.full((K - half,), q_lock_val_b, dtype=torch.float32), atol=1e-5)
print("Test3 held_a:", held_a, " held_b:", held_b)

# --- Test 4 (NEW): regularization actually pulls solution toward q_ref when
#     multiple solutions exist (redundant manipulator, no lock) ---
q_ref4 = q_true.unsqueeze(0).expand(K, -1).clone()  # pull toward the TRUE joint config
q_ref4[:, 0] += 0.5  # perturb q_ref slightly off q_true to see if solution follows it
seeds4 = torch.rand(K, 7, dtype=torch.float32) * (q_hi - q_lo) + q_lo
q_sol4, pos_err4, rot_err4 = solve_batch_ik(target_pos_b, target_rot_b, seeds4, q_lo, q_hi,
                                             no_lock_mask, no_lock_vec, q_ref4, iters=30, lambda_reg=2.0)
best4 = pos_err4.argmin()
dist_to_ref = (q_sol4[best4] - q_ref4[0]).norm().item()
dist_to_qtrue = (q_sol4[best4] - q_true).norm().item()
print("Test4 (regularization pull) pos_err:", pos_err4[best4].item(),
      " dist_to_q_ref:", dist_to_ref, " dist_to_q_true:", dist_to_qtrue)
t4_pass = dist_to_ref < dist_to_qtrue  # solution should be pulled toward q_ref, not just toward q_true

print("=" * 60)
print("t1_pass", t1_pass, " t2_pass", t2_pass, " t2_locked_held", t2_locked_held,
      " held_a", held_a, " held_b", held_b, " t4_pass", t4_pass)
if t1_pass and t2_pass and t2_locked_held and held_a and held_b and t4_pass:
    print("IK_PROJECTOR_VERIFIED")
else:
    sys.exit(1)
