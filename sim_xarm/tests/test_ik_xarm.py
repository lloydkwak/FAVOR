"""The FAVOR IK modules (docker/ik_*.py) run unchanged on XArm7Kinematics.
For configurations around the pouring home pose and a small lock offset on one joint, checks
(1) the faulted joint is held, (2) Priority IK recovers the cup-mouth position, (3) the reversed
order recovers the orientation, (4) the xArm7 limits (not the Panda ones) are applied.
Run: python sim_xarm/tests/test_ik_xarm.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "docker"))
import numpy as np
import torch

from favor_xarm.config import load_config
from favor_xarm.kinematics import from_scene
from favor_xarm.scene import build_scene
from ik_priority import ik_priority
from ik_priority_rev import ik_priority_rev
from ik_redistribution import ik_redistribute, joint_limits

cfg = load_config()
h = build_scene(cfg, n_particles=0)
kin = from_scene(h, cfg)
torch.manual_seed(0)
q0 = torch.tensor(cfg["robot"]["home_q"], dtype=torch.float32)
q = q0 + 0.3 * (torch.rand(64, 7) - 0.5)
lo, hi = joint_limits(kin, "cpu")
ok = bool(torch.allclose(lo, torch.tensor(np.radians([a for a, _ in cfg["robot"]["joint_limits_deg"]]), dtype=torch.float32)))
print(f"xArm7 limits used: {ok}  (J4 lower {np.degrees(lo[3].item()):.1f} deg)")
for j in range(7):
    q_con = q[:, j] + 0.05
    qa, a = ik_priority(kin, q, j, q_con, lam1=0.01, lam2=0.2)
    qb, b = ik_priority_rev(kin, q, j, q_con, lam1=0.01, lam2=0.2)
    qc, c = ik_redistribute(kin, q, j, q_con, rot_weight=0.05, reg=0.1)
    held = max((qa[:, j] - q_con).abs().max().item(), (qb[:, j] - q_con).abs().max().item())
    pa = a["pos_err_after"].median().item() * 1000
    rb = b["rot_err_after"].median().item()
    passed = held < 1e-5 and pa < 1.0 and rb < 0.01 and bool(torch.isfinite(qc).all())
    ok &= passed
    print(f"joint{j + 1}: held {held:.1e}  prio pos {pa:.3f} mm rot {a['rot_err_after'].median().item():.4f} rad | "
          f"rev pos {b['pos_err_after'].median().item() * 1000:.2f} mm rot {rb:.4f} rad | "
          f"W-IK pos {c['pos_err_after'].median().item() * 1000:.2f} mm -> {'PASS' if passed else 'FAIL'}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
