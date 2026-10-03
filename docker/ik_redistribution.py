"""
W-IK baseline: model-based kinematic redistribution applied AFTER sampling.

The policy's sampled joint targets define the intended end-effector pose
(FK, position + orientation). The faulted joint is fixed to its admissible
value (locked: q_onset; range_reduced: clamped into [q_lo, q_hi]) and the
6 healthy joints are re-solved by damped least-squares IK to reach that
intended pose -- the classical fault-tolerant redundancy-resolution
approach, applied to a learned policy's output. No diffusion-time
intervention: the sample is the same one B1 draws (same seed).
"""
import torch

PANDA_Q_LO = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_HI = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def rotvec_from_matrix(R):
    """(..., 3, 3) rotation matrices -> (..., 3) axis*angle vectors."""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    ang = torch.acos(((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    s = torch.sin(ang)
    scale = torch.where(s.abs() < 1e-6, torch.full_like(ang, 0.5), ang / (2.0 * s.clamp_min(1e-6)))
    return v * scale.unsqueeze(-1)


# Chosen a priori from kinematics only (scripts_libero/ik_select_and_layer1v2.py),
# never from rollout outcomes. Overwritten by that script.
IK_ROT_WEIGHT = 0.05
IK_REG = 0.1


def ik_redistribute(kin, q_target, joint_idx, q_con, n_iter=30, rot_weight=None, reg=None,
                    damping=1e-6, max_step=0.2, tol=1e-5):
    """
    Weighted, regularized (Levenberg-Marquardt) IK over the 6 healthy joints:
      min_q  |e_pos|^2 + rot_weight^2 |e_rot|^2 + reg^2 |q_free - q_target_free|^2
    with the faulted joint fixed at q_con. rot_weight trades orientation (rad)
    against position (m); reg keeps the solution near the policy's own targets
    (prevents far-away IK branches / large jumps when the pose is unreachable).
    q_target: (n,7) intended targets; q_con: (n,) admissible faulted-joint value.
    """
    rot_weight = IK_ROT_WEIGHT if rot_weight is None else rot_weight
    reg = IK_REG if reg is None else reg
    dev = q_target.device
    q_target = q_target.to(torch.float32)
    lo, hi = PANDA_Q_LO.to(dev), PANDA_Q_HI.to(dev)
    free = [k for k in range(7) if k != joint_idx]
    w = torch.tensor([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight], device=dev)
    I6 = torch.eye(6, device=dev)
    with torch.no_grad():
        p_t, R_t = kin.forward(q_target)

        def err(q):
            p, R = kin.forward(q)
            return torch.cat([p_t - p, rotvec_from_matrix(R_t @ R.transpose(-1, -2))], dim=-1)

        q = q_target.clone()
        q[:, joint_idx] = q_con.to(dev, torch.float32)
        e0 = err(q)
        for _ in range(n_iter):
            e = err(q)
            Jw = kin.jacobian(q)[:, :, free] * w.view(1, 6, 1)             # (n,6,6)
            g = (Jw.transpose(-1, -2) @ (e * w).unsqueeze(-1)).squeeze(-1) \
                - reg ** 2 * (q[:, free] - q_target[:, free])
            H = Jw.transpose(-1, -2) @ Jw + (reg ** 2 + damping) * I6
            dq = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1).clamp(-max_step, max_step)
            q[:, free] = torch.max(torch.min(q[:, free] + dq, hi[free]), lo[free])
            if dq.abs().max() < tol:
                break
        e1 = err(q)
    info = {
        "pos_err_before": e0[:, :3].norm(dim=-1), "rot_err_before": e0[:, 3:].norm(dim=-1),
        "pos_err_after": e1[:, :3].norm(dim=-1), "rot_err_after": e1[:, 3:].norm(dim=-1),
        "dq_free": (q - q_target)[:, free].abs().max(dim=-1).values,
    }
    return q, info


def summarize_ik_log(policy):
    log = getattr(policy, "ik_log", None)
    if not log:
        return None
    n = sum(d["n"] for d in log)
    wm = lambda k: float(sum(d[k] * d["n"] for d in log) / max(n, 1))
    return {"n_waypoints_adjusted": int(n), "n_calls_with_adjustment": len(log),
            "pos_err_before_mean": wm("pos_before"), "pos_err_after_mean": wm("pos_after"),
            "rot_err_before_mean": wm("rot_before"), "rot_err_after_mean": wm("rot_after"),
            "dq_free_max_mean": wm("dq")}
