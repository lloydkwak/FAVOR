"""
RG-DDPM internal-correction variants (additional runs), separating HOW the internal correction
splits the error from HOW FAR it may move the healthy joints.

Identical to reach_guided.reach_correct (same reachability test, same healthy-chart
re-expression, same epsilon rewrite) except for the step that moves an unreachable waypoint
to a reachable pose on the faulty robot:

  original RG-DDPM   weighted pose IK (w_r 1.0) regularized to the policy posture (reg 0.1)
  internal='prio'    position-first Priority IK (ik_priority.py, lam2 0.2)
  internal='weighted' weighted pose IK (w_r 1.0) without posture term (reg 0, damping 1e-4)
  budget=None        no limit on how far the healthy joints move
  budget=0.3         every healthy joint clamped to within 0.3 rad of the policy's waypoint
                     (the same 0.3 rad that the reachability test uses)

Methods: rg_prioint (prio, no budget), rg_prioint_b03 (prio, budget 0.3),
rg_wint_nob (weighted, no budget).
"""
import functools
import torch
from ik_priority import ik_priority
from reach_guided import ik_lm, _pose_err


def reach_correct_variant(policy, x0, kin, j, q_lo_phys, q_hi_phys, cfg, state, acc,
                          internal="prio", budget=None, iters=10):
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
        if internal == "prio":
            u_b, _ = ik_priority(kin, q[idx], j, q_con[idx], n_iter=iters, lam2=0.2)
        else:
            u_b = ik_lm(kin, p_t[idx], R_t[idx], q_ref_f[idx].clone(), q_ref_f[idx], free_f,
                        iters, 1.0, 0.0, damping=1e-4)
        if budget is not None:
            u_b = u_b.clone()
            u_b[:, free_f] = q_ref_f[idx][:, free_f] + (u_b - q_ref_f[idx])[:, free_f].clamp(-budget, budget)
        p_r, R_r = kin.forward(u_b)
        q_h = ik_lm(kin, p_r, R_r, q[idx].clone(), q[idx], list(range(7)),
                    cfg.iters_h, cfg.rot_weight, cfg.reg)
        acc['mod_sum'] = acc.get('mod_sum', 0.0) + float((q_h - q[idx]).abs().max(dim=-1).values.sum())
        q_out = q.clone(); q_out[idx] = q_h
    a2 = a.clone()
    a2[..., :7] = q_out.reshape(B, T, 7).to(a2.dtype)
    mask = torch.zeros(B * T, dtype=torch.bool, device=q.device); mask[idx] = True
    return nz.normalize(a2), mask.reshape(B, T)


reach_correct_prio = functools.partial(reach_correct_variant, internal="prio", budget=None)
VARIANTS = {
    "rg_prioint": dict(internal="prio", budget=None),
    "rg_prioint_b03": dict(internal="prio", budget=0.3),
    "rg_wint_nob": dict(internal="weighted", budget=None),
}
