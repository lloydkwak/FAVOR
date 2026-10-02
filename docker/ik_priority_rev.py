"""
Reversed task-priority IK (ablation of ik_priority.py): ORIENTATION is the 1st task and
POSITION is corrected only inside the null space of the orientation task. Same damping
(lam1 on the 1st task, lam2 on the 2nd), iterations and limits as ik_priority, so the
only change is the order of the two tasks.
"""
import torch
from ik_redistribution import rotvec_from_matrix, PANDA_Q_LO, PANDA_Q_HI
from ik_priority import _dpinv


def ik_priority_rev(kin, q_target, joint_idx, q_con, n_iter=30, lam1=0.01, lam2=0.1,
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
            Jr_pinv = _dpinv(Jr, lam1)
            dq1 = (Jr_pinv @ er.unsqueeze(-1)).squeeze(-1)                 # 1st: orientation
            N1 = I6 - Jr_pinv @ Jr
            res_p = ep - (Jp @ dq1.unsqueeze(-1)).squeeze(-1)
            dq2 = (_dpinv(Jp @ N1, lam2) @ res_p.unsqueeze(-1)).squeeze(-1)  # 2nd: position
            dq = (dq1 + (N1 @ dq2.unsqueeze(-1)).squeeze(-1)).clamp(-max_step, max_step)
            q[:, free] = torch.max(torch.min(q[:, free] + dq, hi[free]), lo[free])
            if dq.abs().max() < tol:
                break
        ep1, er1 = err(q)
    info = {"pos_err_before": ep0.norm(dim=-1), "rot_err_before": er0.norm(dim=-1),
            "pos_err_after": ep1.norm(dim=-1), "rot_err_after": er1.norm(dim=-1),
            "dq_free": (q - q_target)[:, free].abs().max(dim=-1).values}
    return q, info
