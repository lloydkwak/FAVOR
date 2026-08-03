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


def solve_batch_ik(target_pos, target_rot, q_seed_batch, q_lo, q_hi, locked_mask, q_lock, iters, damping=1e-3):
    """
    Vectorized over an arbitrary leading batch dim N (e.g. N = n_envs * K).
    target_pos: (N,3)  target_rot: (N,3,3)  q_seed_batch: (N,7)
    q_lo/q_hi/q_lock: (7,)  locked_mask: (7,) bool
    Uses the FULL 6D (position+orientation) error with the closed-form 6x7
    Jacobian -- previously only the position rows of J were used, so
    orientation error never actually influenced the Newton step.
    """
    N = q_seed_batch.shape[0]
    q = q_seed_batch.clone()
    for _ in range(iters):
        J, cur_pos, cur_rot = panda_jacobian(q)          # J: (N,6,7)
        pos_err = target_pos - cur_pos
        rot_err = rotmat_to_axis_angle(target_rot @ cur_rot.transpose(-1, -2))
        e = torch.cat([pos_err, rot_err], dim=-1)         # (N,6)

        J_full = J
        if locked_mask.any():
            J_full = J_full.clone()
            J_full[:, :, locked_mask] = 0.0

        JJt = J_full @ J_full.transpose(-1, -2)           # (N,6,6)
        lam_I = damping * torch.eye(6, device=q.device, dtype=q.dtype).unsqueeze(0)
        inv = torch.linalg.solve(JJt + lam_I, e.unsqueeze(-1)).squeeze(-1)  # (N,6)
        dq = torch.einsum('nij,ni->nj', J_full, inv)      # (N,7)

        q = q + dq
        q = torch.clamp(q, q_lo, q_hi)
        if locked_mask.any():
            q[:, locked_mask] = q_lock.expand(N, -1)[:, locked_mask]

    J, final_pos, final_rot = panda_jacobian(q)
    pos_err = (target_pos - final_pos).norm(dim=-1)
    rot_err = rotmat_to_axis_angle(target_rot @ final_rot.transpose(-1, -2)).norm(dim=-1)
    return q, pos_err, rot_err


def _rot6d_to_matrix(rot6):
    a1, a2 = rot6[..., 0:3], rot6[..., 3:6]
    a1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a2 = a2 - (a1 * a2).sum(dim=-1, keepdim=True) * a1
    a2 = a2 / a2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-1)


def _matrix_to_rot6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def _project_waypoints_impl(raw_clean, fault_spec, q_prev_seed, K, iters):
    """
    raw_clean: (B, Tp, Da) on ANY device -- moved to CPU internally (IK is
    CPU-bound per hardware spec), moved back to original device on return.
    Vectorizes over (B*K) within each waypoint; loops only over Tp (16),
    not B*Tp (80), preserving the sequential warm-start chain across
    waypoints (each waypoint's seed = previous waypoint's chosen solution).
    """
    orig_device = raw_clean.device
    dtype = torch.float32
    cpu = torch.device("cpu")

    raw_clean_cpu = raw_clean.detach().to(cpu, dtype=dtype)
    B, Tp, Da = raw_clean_cpu.shape

    joint_idx = fault_spec['joint_idx']
    q_lock_scalar = fault_spec['q_lock']
    q_lo = fault_spec['q_lo'].to(cpu, dtype=dtype)
    q_hi = fault_spec['q_hi'].to(cpu, dtype=dtype)
    is_locked = fault_spec['fault_type'] == 'locked'

    locked_mask = torch.zeros(7, dtype=torch.bool)
    locked_mask[joint_idx] = is_locked
    q_lock_vec = torch.zeros(7, dtype=dtype)
    if is_locked:
        q_lock_vec[joint_idx] = q_lock_scalar

    corrected = raw_clean_cpu.clone()
    q_seed = q_prev_seed.detach().to(cpu, dtype=dtype).clone()  # (7,) or (B,7)
    if q_seed.dim() == 1:
        q_seed = q_seed.unsqueeze(0).expand(B, -1).clone()

    for k in range(Tp):
        pos_k = raw_clean_cpu[:, k, 0:3]                          # (B,3)
        rot6_k = raw_clean_cpu[:, k, 3:9]
        target_rot_k = _rot6d_to_matrix(rot6_k)                   # (B,3,3)

        seeds = torch.rand(B, K, 7, dtype=dtype) * (q_hi - q_lo) + q_lo
        seeds[:, 0, :] = q_seed                                    # warm start, per-env
        if locked_mask.any():
            seeds[:, :, joint_idx] = q_lock_scalar

        N = B * K
        seeds_flat = seeds.reshape(N, 7)
        target_pos_flat = pos_k.unsqueeze(1).expand(B, K, 3).reshape(N, 3)
        target_rot_flat = target_rot_k.unsqueeze(1).expand(B, K, 3, 3).reshape(N, 3, 3)

        q_sol_flat, pos_err_flat, rot_err_flat = solve_batch_ik(
            target_pos_flat, target_rot_flat, seeds_flat, q_lo, q_hi,
            locked_mask, q_lock_vec, iters)

        q_sol = q_sol_flat.reshape(B, K, 7)
        pos_err = pos_err_flat.reshape(B, K)
        rot_err = rot_err_flat.reshape(B, K)

        converged = (pos_err < 0.01) & (rot_err < 0.15)
        new_seed = torch.zeros(B, 7, dtype=dtype)
        for b in range(B):
            if converged[b].any():
                cand = q_sol[b][converged[b]]
                dists = (cand - q_seed[b].unsqueeze(0)).norm(dim=-1)
                best = cand[dists.argmin()]
            else:
                best = q_sol[b][pos_err[b].argmin()]
            new_seed[b] = best
        q_seed = new_seed

        fk_pos, fk_rot = panda_fk(q_seed)
        corrected[:, k, 0:3] = fk_pos
        corrected[:, k, 3:9] = _matrix_to_rot6d(fk_rot)

    return corrected.to(orig_device, dtype=raw_clean.dtype), q_seed.to(orig_device, dtype=raw_clean.dtype)


def project_waypoints(raw_clean, fault_spec, q_prev_seed, K=64, iters=5):
    corrected, _ = _project_waypoints_impl(raw_clean, fault_spec, q_prev_seed, K, iters)
    return corrected


class ProjectWaypoints:
    """Stateful wrapper so callers can read the final joint seed after a call
    (favor_policy.py uses this to carry warm-start continuity across steps).
    last_q_seed is now (B,7) -- one seed per environment, not a single (7,)
    vector -- since IK is vectorized across the batch."""
    def __init__(self, K=64, iters=5):
        self.K = K
        self.iters = iters
        self.last_q_seed = None

    def __call__(self, raw_clean, fault_spec, q_prev_seed):
        corrected, last_seed = _project_waypoints_impl(raw_clean, fault_spec, q_prev_seed, self.K, self.iters)
        self.last_q_seed = last_seed
        return corrected
