"""
RG-DDPM with a prioritized internal correction (revision variant 'rg_prioint').

Identical to reach_guided.reach_correct (same reachability test, same healthy-chart
re-expression, same epsilon rewrite) except for the step that moves an unreachable
waypoint to a reachable pose: instead of weighted pose IK limited to the execution
layer's motion budget, the faulty robot's pose is found by position-first Priority IK
(ik_priority.py, lam2 = 0.2), with no motion limit. This fills the missing cell of the
where x how comparison: denoising-time guidance whose internal correction is prioritized.
"""
import torch
from ik_priority import ik_priority
from reach_guided import ik_lm, _pose_err


def reach_correct_prio(policy, x0, kin, j, q_lo_phys, q_hi_phys, cfg, state, acc, prio_iters=10):
    B, T, Da = x0.shape
    nz = policy.normalizer['action']
    a = nz.unnormalize(x0)
    q = a[..., :7].reshape(B * T, 7).float()
    lo = q_lo_phys[..., j].to(q.device).float().reshape(-1, 1).expand(B, T).reshape(-1)
    hi = q_hi_phys[..., j].to(q.device).float().reshape(-1, 1).expand(B, T).reshape(-1)
    q_con = torch.max(torch.min(q[:, j], hi), lo)
    with torch.no_grad():
        p_t, R_t = kin.forward(q)
        free_f = [k for k in range(7) if k != j]
        q_ref_f = q.clone(); q_ref_f[:, j] = q_con
        u_init = q_ref_f if state.get('u') is None else state['u'].clone()
        u_init[:, j] = q_con
        u = ik_lm(kin, p_t, R_t, u_init, q_ref_f, free_f, cfg.iters_f, cfg.rot_weight, cfg.reg_f)
        state['u'] = u
        pe, re = _pose_err(kin, u, p_t, R_t)
        big = (u - q_ref_f)[:, free_f].abs().max(dim=-1).values > cfg.dq_max
        unreach = (pe > cfg.tol_pos) | (re > cfg.tol_rot) | big
        acc['n_checked'] = acc.get('n_checked', 0) + unreach.numel()
        acc['n_unreach'] = acc.get('n_unreach', 0) + int(unreach.sum())
        if not unreach.any():
            return None
        idx = unreach.nonzero().squeeze(-1)
        u_b, _ = ik_priority(kin, q[idx], j, q_con[idx], n_iter=prio_iters, lam2=0.2)   # <- prioritized
        p_r, R_r = kin.forward(u_b)
        q_h = ik_lm(kin, p_r, R_r, q[idx].clone(), q[idx], list(range(7)),
                    cfg.iters_h, cfg.rot_weight, cfg.reg)
        acc['mod_sum'] = acc.get('mod_sum', 0.0) + float((q_h - q[idx]).abs().max(dim=-1).values.sum())
        q_out = q.clone(); q_out[idx] = q_h
    a2 = a.clone()
    a2[..., :7] = q_out.reshape(B, T, 7).to(a2.dtype)
    mask = torch.zeros(B * T, dtype=torch.bool, device=q.device); mask[idx] = True
    return nz.normalize(a2), mask.reshape(B, T)
