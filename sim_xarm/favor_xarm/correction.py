"""Post-hoc fault correction of a policy action chunk on the xArm7, with the semantics of
docker/native_joint_policy*.py: on the executed waypoints, the faulted joint's admissible value is
q_con = clip(q*_j, lo, hi); only waypoints where q_con differs from the policy value are re-solved
over the six healthy joints. Method parameters are the paper's, fixed (never tuned here).

  B1                         : no correction (the fault layer alone holds the joint)
  W-IK pos / W-IK pose       : ik_redistribute, w_r 0.05 / 1.0, rho 0.1, mu 1e-6, 30 it
  Priority IK                : ik_priority, lambda_1 0.01, lambda_2 0.2, 30 it
  Priority IK (orient. first): ik_priority_rev, same damping
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docker"))
from ik_priority import ik_priority                     # noqa: E402
from ik_priority_rev import ik_priority_rev             # noqa: E402
from ik_redistribution import ik_redistribute           # noqa: E402

METHODS = {
    "B1": None,
    "W-IK pos": lambda kin, q, j, c: ik_redistribute(kin, q, j, c, n_iter=30, rot_weight=0.05, reg=0.1, damping=1e-6),
    "W-IK pose": lambda kin, q, j, c: ik_redistribute(kin, q, j, c, n_iter=30, rot_weight=1.0, reg=0.1, damping=1e-6),
    "Priority IK": lambda kin, q, j, c: ik_priority(kin, q, j, c, n_iter=30, lam1=0.01, lam2=0.2),
    "Priority IK (orient. first)": lambda kin, q, j, c: ik_priority_rev(kin, q, j, c, n_iter=30, lam1=0.01, lam2=0.2),
}


def correct_chunk(kin, q_exec, fault_info, method):
    """q_exec: (K, 7) joint targets to execute. fault_info: FaultLayer.get_fault_info().
    Returns (corrected (K, 7) tensor, log dict or None)."""
    q_exec = torch.as_tensor(q_exec, dtype=torch.float32).clone()
    if METHODS[method] is None or fault_info is None or fault_info.get("fault_type") is None:
        return q_exec, None
    j = fault_info["joint_idx"]
    q_con = q_exec[:, j].clamp(float(fault_info["lo"]), float(fault_info["hi"]))
    need = (q_con - q_exec[:, j]).abs() > 1e-6
    if not need.any():
        return q_exec, None
    idx = need.nonzero().squeeze(-1)
    q_new, info = METHODS[method](kin, q_exec[idx], j, q_con[idx])
    q_exec[idx] = q_new
    return q_exec, {"n": int(len(idx)), "pos_after_mm": 1000 * float(info["pos_err_after"].mean()),
                    "rot_after_deg": float(torch.rad2deg(info["rot_err_after"].mean())),
                    "dq": float(info["dq_free"].mean())}
