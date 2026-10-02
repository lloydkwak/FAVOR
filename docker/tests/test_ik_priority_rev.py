"""
Offline check of the priority ORDER: on locked-joint offsets, ik_priority (position first)
must keep the EE position, ik_priority_rev (orientation first) must keep the orientation.
Run inside the container:  python /workspace/docker/tests/test_ik_priority_rev.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "/workspace/docker")
import torch
from fault_kinematics import PandaKinematics
from ik_priority import ik_priority
from ik_priority_rev import ik_priority_rev

torch.manual_seed(0)
kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.zeros(3), torch.eye(3))
q0 = torch.tensor([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8])
q = q0 + 0.3 * (torch.rand(64, 7) - 0.5)
ok = True
for j, off in [(0, 0.05), (5, 0.3), (6, 0.5)]:
    q_con = q[:, j] + off
    _, a = ik_priority(kin, q, j, q_con, lam2=0.2)
    _, b = ik_priority_rev(kin, q, j, q_con, lam2=0.2)
    pa, ra = a["pos_err_after"].median().item() * 1000, a["rot_err_after"].median().item()
    pb, rb = b["pos_err_after"].median().item() * 1000, b["rot_err_after"].median().item()
    passed = pa < 1.0 and rb < 0.01
    ok &= passed
    print(f"joint{j + 1} offset {off}: pos-first pos={pa:6.2f} mm rot={ra:.4f} rad | "
          f"rot-first pos={pb:6.2f} mm rot={rb:.4f} rad -> {'PASS' if passed else 'FAIL'}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
