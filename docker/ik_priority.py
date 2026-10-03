"""
Task-priority re-embodiment IK (Nakamura 1987; Siciliano & Slotine 1991): with the
faulted joint fixed, the 6 healthy joints first restore the intended EE POSITION;
orientation is corrected only inside the null space of the position task, so it
can never be bought with position error. Motivation (offline analysis of W-IK):
full-pose IK fails on j6/j7 because it gives up 4-8 cm of position to fix
orientation, while position-only IK leaves ~3 deg tilt on j1/j3 that the null
space could have removed.
"""
import torch
from ik_redistribution import rotvec_from_matrix, PANDA_Q_LO, PANDA_Q_HI


def _dpinv(A, lam):
    m = A.shape[-2]
    I = torch.eye(m, device=A.device, dtype=A.dtype)
    return A.transpose(-1, -2) @ torch.linalg.inv(A @ A.transpose(-1, -2) + lam ** 2 * I)


def ik_priority(kin, q_target, joint_idx, q_con, n_iter=30, lam1=0.01, lam2=0.1,
                max_step=0.2, tol=1e-6):
    dev = q_target.device
    q_target = q_target.float()
    lo, hi = PANDA_Q_LO.to(dev), PANDA_Q_HI.to(dev)
    free = [k for k in range(7) if k != joint_idx]
    I6 = torch.eye(6, device=dev)
    with torch.no_grad():
        p_t, R_t = kin.forward(q_target)

        def err(q):
            p, R = kin.forward(q)
            return p_t - p, rotvec_from_matrix(R_t @ R.transpose(-1, -2))

        q = q_target.clone(); q[:, joint_idx] = q_con.to(dev, torch.float32)
        ep0, er0 = err(q)
        for _ in range(n_iter):
            ep, er = err(q)
            J = kin.jacobian(q)[:, :, free]
            Jp, Jr = J[:, :3], J[:, 3:]
            Jp_pinv = _dpinv(Jp, lam1)
            dq1 = (Jp_pinv @ ep.unsqueeze(-1)).squeeze(-1)
            N1 = I6 - Jp_pinv @ Jp
            res_r = er - (Jr @ dq1.unsqueeze(-1)).squeeze(-1)
            dq2 = (_dpinv(Jr @ N1, lam2) @ res_r.unsqueeze(-1)).squeeze(-1)
            dq = (dq1 + (N1 @ dq2.unsqueeze(-1)).squeeze(-1)).clamp(-max_step, max_step)
            q[:, free] = torch.max(torch.min(q[:, free] + dq, hi[free]), lo[free])
            if dq.abs().max() < tol:
                break
        ep1, er1 = err(q)
    info = {"pos_err_before": ep0.norm(dim=-1), "rot_err_before": er0.norm(dim=-1),
            "pos_err_after": ep1.norm(dim=-1), "rot_err_after": er1.norm(dim=-1),
            "dq_free": (q - q_target)[:, free].abs().max(dim=-1).values}
    return q, info
