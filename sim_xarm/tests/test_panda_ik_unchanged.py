"""Regression guard for the joint-limit hook added to docker/ik_*.py: the Panda results must stay
bit-identical. Runs W-IK (pos, pose, rho 0), Priority IK and reversed Priority IK on fixed random
Panda inputs for every joint and compares a SHA-256 of all outputs with the hash recorded from the
unmodified modules (same torch build; a different torch/BLAS build may change the float bits, in
which case re-record the hash on the pre-change commit with --record).
Run: python sim_xarm/tests/test_panda_ik_unchanged.py"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docker"))
import torch
from fault_kinematics import PandaKinematics
from ik_priority import ik_priority
from ik_priority_rev import ik_priority_rev
import ik_redistribution as ikr

# Recorded on commit 42d985a (modules before the hook), torch 2.15.0 CPU.
EXPECTED = "3a09d61c5c728f0bdda0ee74c177cc40fd3c7f61810b364babfc8666d36980f7"

torch.manual_seed(0)
kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.tensor([-0.6, 0, 0]), torch.eye(3))
q0 = torch.tensor([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8])
q = q0 + 0.6 * (torch.rand(128, 7) - 0.5)
out = {}
for j in range(7):
    c = q[:, j] + 0.1
    out[("prio", j)] = ik_priority(kin, q, j, c, lam1=0.01, lam2=0.2)
    out[("rev", j)] = ik_priority_rev(kin, q, j, c, lam1=0.01, lam2=0.2)
    out[("wpos", j)] = ikr.ik_redistribute(kin, q, j, c, rot_weight=0.05, reg=0.1)
    out[("wpose", j)] = ikr.ik_redistribute(kin, q, j, c, rot_weight=1.0, reg=0.1)
    out[("wik0", j)] = ikr.ik_redistribute(kin, q, j, c, rot_weight=0.05, reg=0.0, damping=1e-4)
h = hashlib.sha256()
for k in sorted(out, key=str):
    qq, info = out[k]
    h.update(qq.numpy().tobytes())
    for x in sorted(info):
        h.update(info[x].numpy().tobytes())
got = h.hexdigest()
if "--record" in sys.argv:
    print(got)
    sys.exit(0)
print("hash", got, "PASS" if got == EXPECTED else "FAIL (expected %s)" % EXPECTED)
sys.exit(0 if got == EXPECTED else 1)
