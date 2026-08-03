import torch
from panda_kinematics import panda_fk, panda_jacobian

def rotmat_to_axis_angle(R):
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)
    denom = 2 * torch.sin(theta).clamp(min=1e-6).unsqueeze(-1)
    return axis / denom * theta.unsqueeze(-1)

def solve_batch_ik(target_pos, target_rot, q_seed_batch, q_lo, q_hi, locked_mask, iters, damping=1e-3):
    K = q_seed_batch.shape[0]
    q = q_seed_batch.clone()
    for _ in range(iters):
        J, cur_pos, cur_rot = panda_jacobian(q)
        pos_err = target_pos.unsqueeze(0).expand(K, -1) - cur_pos
        rot_err = rotmat_to_axis_angle(target_rot.unsqueeze(0).expand(K, -1, -1) @ cur_rot.transpose(-1, -2))
        e = torch.cat([pos_err, rot_err], dim=-1)
        J_full = J
        if locked_mask.any():
            J_full = J_full.clone()
            J_full[:, :, locked_mask] = 0.0
        JJt = J_full @ J_full.transpose(-1, -2)
        lam_I = damping * torch.eye(3, device=q.device, dtype=q.dtype).unsqueeze(0)
        inv = torch.linalg.solve(JJt[:, :3, :3] + lam_I, e[:, :3].unsqueeze(-1)).squeeze(-1)
        dq = torch.einsum('kij,ki->kj', J_full[:, :3, :], inv)
        q = q + dq
        q = torch.clamp(q, q_lo.unsqueeze(0), q_hi.unsqueeze(0))
        if locked_mask.any():
            q[:, locked_mask] = q_lo.unsqueeze(0).expand(K, -1)[:, locked_mask]
    J, final_pos, final_rot = panda_jacobian(q)
    pos_err = (target_pos.unsqueeze(0).expand(K, -1) - final_pos).norm(dim=-1)
    return q, pos_err

def project_waypoints(raw_clean, fault_spec, q_prev_seed, K=64, iters=5):
    B, Tp, Da = raw_clean.shape
    device = raw_clean.device
    dtype = raw_clean.dtype
    joint_idx = fault_spec['joint_idx']
    q_lock = fault_spec['q_lock']
    q_lo = fault_spec['q_lo'].to(device=device, dtype=dtype)
    q_hi = fault_spec['q_hi'].to(device=device, dtype=dtype)
    locked_mask = torch.zeros(7, dtype=torch.bool, device=device)
    locked_mask[joint_idx] = fault_spec['fault_type'] == 'locked'

    corrected = raw_clean.clone()
    q_seed = q_prev_seed.clone()

    for b in range(B):
        for k in range(Tp):
            pos = raw_clean[b, k, 0:3]
            rot6 = raw_clean[b, k, 3:9]
            a1, a2 = rot6[0:3], rot6[3:6]
            a1 = a1 / a1.norm().clamp(min=1e-6)
            a2 = a2 - (a1 * a2).sum() * a1
            a2 = a2 / a2.norm().clamp(min=1e-6)
            a3 = torch.cross(a1, a2)
            target_rot = torch.stack([a1, a2, a3], dim=-1)

            seeds = torch.rand(K, 7, device=device, dtype=dtype) * (q_hi - q_lo) + q_lo
            seeds[0] = q_seed
            if locked_mask.any():
                seeds[:, joint_idx] = q_lock

            q_sol, pos_err = solve_batch_ik(pos, target_rot, seeds, q_lo, q_hi, locked_mask, iters)
            converged = pos_err < 0.01
            if converged.any():
                dists = (q_sol[converged] - q_seed.unsqueeze(0)).norm(dim=-1)
                best = q_sol[converged][dists.argmin()]
            else:
                best = q_sol[pos_err.argmin()]

            fk_pos, fk_rot = panda_fk(best)
            corrected[b, k, 0:3] = fk_pos
            r0, r1 = fk_rot[:, 0], fk_rot[:, 1]
            corrected[b, k, 3:9] = torch.cat([r0, r1])
            q_seed = best.detach()

    return corrected
