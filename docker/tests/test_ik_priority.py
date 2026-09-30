"""
Unit test for Priority IK (docker/ik_priority.py).
For random Panda configurations and a small lock offset on one joint, checks that
  (1) the faulted joint is held exactly at the commanded value,
  (2) the end-effector position is recovered (1st priority),
  (3) for shoulder/elbow faults the orientation is recovered too (2nd priority).
Run inside the container:  python /workspace/docker/tests/test_ik_priority.py
"""
import sys
sys.path.insert(0, "/workspace/docker")
import torch
from fault_kinematics import PandaKinematics
from ik_priority import ik_priority

torch.manual_seed(0)
kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.zeros(3), torch.eye(3))
q0 = torch.tensor([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8])
q = q0 + 0.3 * (torch.rand(64, 7) - 0.5)

ok = True
for j, check_rot in [(0, True), (2, True), (4, True), (5, False), (6, False)]:
    q_con = q[:, j] + 0.05
    q_new, info = ik_priority(kin, q, j, q_con, lam2=0.2)
    held = (q_new[:, j] - q_con).abs().max().item()
    pos = info["pos_err_after"].median().item()
    rot = info["rot_err_after"].median().item()
    passed = held < 1e-5 and pos < 1e-3 and (rot < 0.02 or not check_rot)
    ok &= passed
    print(f"joint{j + 1}: held_err={held:.1e}  median pos_err={pos * 1000:.3f} mm  "
          f"median rot_err={rot:.4f} rad  -> {'PASS' if passed else 'FAIL'}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
