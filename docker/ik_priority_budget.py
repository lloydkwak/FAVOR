"""Motion-budgeted task-priority IK (wraps ik_priority; does not modify it).
q_pos  = priority IK with orientation task effectively off (lam2 huge)  -> position-exact, min motion
q_full = priority IK with lam2 (default 0.2)                            -> position + null-space orientation
q      = q_pos + alpha*(q_full - q_pos),  alpha = min(1, budget/||q_full - q_pos||)   (per waypoint)
Safety: if ||q_pos - q_target|| > cap  -> leave waypoint unchanged (no correction).
"""
import torch
from ik_priority import ik_priority

def ik_priority_budget(kin, q_target, joint_idx, q_con, budget=0.3, cap=1.5,
                       lam2=0.2, lam2_off=1e3, **kw):
    q_pos, info_pos = ik_priority(kin, q_target, joint_idx, q_con, lam2=lam2_off, **kw)
    q_full, info = ik_priority(kin, q_target, joint_idx, q_con, lam2=lam2, **kw)
    d = q_full - q_pos
    n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    alpha = torch.clamp(budget / n, max=1.0)
    q = q_pos + alpha * d
    over = (q_pos - q_target).norm(dim=-1, keepdim=True) > cap
    q = torch.where(over, q_target, q)
    free = [j for j in range(q.shape[-1]) if j != joint_idx]
    if 'dq_free' in info:
        info['dq_free'] = (q - q_target)[..., free]
    info['alpha'] = alpha.squeeze(-1)
    info['capped'] = over.squeeze(-1)
    info['dq_total'] = (q - q_target)[..., free].norm(dim=-1)
    return q, info
