"""
RG-DDPM: Reachability-Guided DDPM sampling in the HEALTHY robot's joint chart.

Transplants inference-time cross-embodiment adaptation (EmbodiSteer,
optimization-guided diffusion) to a JOINT-space policy, with the fault expressed
as a new robot model (locked joint = fixed joint, range fault = narrowed limits):
  * the diffusion state always stays in the healthy robot's joint coordinates --
    faulty-body configurations are never injected (the E-C-I / TP failure mode);
  * the constraint is on the TASK outcome, FK(q) in R_f (reachable set of the
    faulty robot), NOT on q in Q_f;
  * where the intended EE trajectory is reachable by the faulty robot, nothing is
    changed: the sampler is bit-identical to B1 (the scheduler's own DDPM step);
  * only unreachable waypoints get their task content moved to the nearest
    reachable pose y', re-expressed in the healthy chart by 7-joint IK regularized
    to x0_hat, and injected by rewriting the epsilon prediction so that the
    scheduler's DDPM step uses the corrected x0.
Execution boundary (in the policy wrapper): final chunk -> faulty IK (= B-IK).
"""
import torch
from ik_redistribution import rotvec_from_matrix, PANDA_Q_LO, PANDA_Q_HI


class RGConfig:
    def __init__(self, start_frac=0.5, iters_f=3, iters_h=3, rot_weight=1.0, reg=0.1,
                 tol_pos=0.003, tol_rot=0.035, reg_f=0.01, dq_max=0.3, iters_b=5):
        # dq_max: 'reachable' = reachable WITHOUT moving the healthy joints more than
        # dq_max (j1/j3 need ~0.12-0.2 rad, j6/j7 0.4-1.0 rad offline); larger
        # compensations are what made B-IK-pose fail on j6/j7, so those intentions
        # are treated as unreachable and re-planned in the healthy chart.
        # reg_f: reachability test only -- must be ~unregularized, otherwise truly
        # reachable poses stop short (~3 mm / 3 deg at reg=0.1) and get flagged.
        self.start_frac, self.iters_f, self.iters_h = start_frac, iters_f, iters_h
        self.rot_weight, self.reg, self.reg_f = rot_weight, reg, reg_f
        self.dq_max, self.iters_b = dq_max, iters_b
        self.tol_pos, self.tol_rot = tol_pos, tol_rot


def ik_lm(kin, p_t, R_t, q_init, q_ref, free, n_iter, rot_weight, reg, damping=1e-6, max_step=0.2):
    dev = q_init.device
    lo, hi = PANDA_Q_LO.to(dev), PANDA_Q_HI.to(dev)
    w = torch.tensor([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight], device=dev)
    I = torch.eye(len(free), device=dev)
    q = q_init.clone().float()
    with torch.no_grad():
        for _ in range(n_iter):
            p, R = kin.forward(q)
            e = torch.cat([p_t - p, rotvec_from_matrix(R_t @ R.transpose(-1, -2))], dim=-1)
            Jw = kin.jacobian(q)[:, :, free] * w.view(1, 6, 1)
            g = (Jw.transpose(-1, -2) @ (e * w).unsqueeze(-1)).squeeze(-1) \
                - reg ** 2 * (q[:, free] - q_ref[:, free])
            H = Jw.transpose(-1, -2) @ Jw + (reg ** 2 + damping) * I
            dq = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1).clamp(-max_step, max_step)
            q[:, free] = torch.max(torch.min(q[:, free] + dq, hi[free]), lo[free])
    return q


def _pose_err(kin, q, p_t, R_t):
    p, R = kin.forward(q)
    return (p_t - p).norm(dim=-1), rotvec_from_matrix(R_t @ R.transpose(-1, -2)).norm(dim=-1)


def reach_correct(policy, x0, kin, j, q_lo_phys, q_hi_phys, cfg, state, acc):
    """x0: (B,T,Da) normalized clean estimate (healthy chart). Returns (x0_new, mask)
    or None if every waypoint is reachable by the faulty robot."""
    B, T, Da = x0.shape
    nz = policy.normalizer['action']
    a = nz.unnormalize(x0)
    q = a[..., :7].reshape(B * T, 7).float()
    lo = q_lo_phys[..., j].to(q.device).float().reshape(-1, 1).expand(B, T).reshape(-1)
    hi = q_hi_phys[..., j].to(q.device).float().reshape(-1, 1).expand(B, T).reshape(-1)
    q_con = torch.max(torch.min(q[:, j], hi), lo)
    with torch.no_grad():
        p_t, R_t = kin.forward(q)                              # intended EE (healthy robot)
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
        # nearest pose reachable within the execution layer's motion budget
        u_b = ik_lm(kin, p_t[idx], R_t[idx], q_ref_f[idx].clone(), q_ref_f[idx], free_f,
                    cfg.iters_b, cfg.rot_weight, cfg.reg)
        p_r, R_r = kin.forward(u_b)
        q_h = ik_lm(kin, p_r, R_r, q[idx].clone(), q[idx], list(range(7)),
                    cfg.iters_h, cfg.rot_weight, cfg.reg)       # healthy-chart re-expression
        acc['mod_sum'] = acc.get('mod_sum', 0.0) + float((q_h - q[idx]).abs().max(dim=-1).values.sum())
        q_out = q.clone(); q_out[idx] = q_h
    a2 = a.clone()
    a2[..., :7] = q_out.reshape(B, T, 7).to(a2.dtype)
    mask = torch.zeros(B * T, dtype=torch.bool, device=q.device); mask[idx] = True
    return nz.normalize(a2), mask.reshape(B, T)


def rg_ddpm_sample(policy, condition_data, condition_mask, kin, joint_idx, q_lo_phys, q_hi_phys,
                   global_cond=None, local_cond=None, generator=None, cfg=None, stats=None):
    cfg = cfg or RGConfig()
    model, sch = policy.model, policy.noise_scheduler
    assert sch.config.prediction_type == 'epsilon', sch.config.prediction_type
    traj = torch.randn(size=condition_data.shape, dtype=condition_data.dtype,
                       device=condition_data.device, generator=generator)
    sch.set_timesteps(policy.num_inference_steps)
    n = len(sch.timesteps)
    start = int(round(cfg.start_frac * n))
    acp = sch.alphas_cumprod.to(traj.device)
    state, acc = {}, {}
    for i, t in enumerate(sch.timesteps):
        traj[condition_mask] = condition_data[condition_mask]
        eps = model(traj, t, local_cond=local_cond, global_cond=global_cond)
        if i >= start:
            ab = acp[t]
            x0 = (traj - (1 - ab).sqrt() * eps) / ab.sqrt()
            if sch.config.clip_sample:
                x0 = x0.clamp(-1.0, 1.0)
            res = reach_correct(policy, x0, kin, joint_idx, q_lo_phys, q_hi_phys, cfg, state, acc)
            if res is not None:
                x0_new, mask = res
                eps_new = (traj - ab.sqrt() * x0_new) / (1 - ab).sqrt()
                eps = torch.where(mask.unsqueeze(-1).expand_as(eps), eps_new, eps)
        traj = sch.step(eps, t, traj, generator=generator).prev_sample
    traj[condition_mask] = condition_data[condition_mask]
    if stats is not None:
        nc = max(acc.get('n_checked', 0), 1)
        stats['frac_unreach'] = acc.get('n_unreach', 0) / nc
        stats['mod_rad'] = acc.get('mod_sum', 0.0) / max(acc.get('n_unreach', 0), 1)
    return traj
