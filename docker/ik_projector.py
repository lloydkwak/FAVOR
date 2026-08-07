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


def solve_batch_ik(target_pos, target_rot, q_seed_batch, q_lo, q_hi, locked_mask, q_lock,
                    q_ref, iters, damping=1e-4, lambda_reg=0.5):
    """lambda_reg may be a python scalar (broadcast to all N) or a (N,) tensor
    (per-sample regularization strength -- used by the two-population
    explore/continuity seed strategy in _project_waypoints_impl)."""
    """
    Vectorized over an arbitrary leading batch dim N.
    NEW: joint-space regularized Gauss-Newton (was task-space DLS with no
    joint-space preference at all). Solves:
        dq* = argmin_dq ||J dq - e||^2 + lambda_reg * ||q + dq - q_ref||^2
    i.e. minimize pose error while staying close to q_ref (the PREVIOUS
    waypoint's chosen solution) -- not just "any solution that happens to
    satisfy the pos/rot tolerance". This directly targets the failure mode
    found empirically: independently-solved waypoints could jump to
    unrelated arm configurations (different elbow posture etc.), causing
    pos_err to grow with waypoint distance (0.005 -> 0.124 over 16
    waypoints) even though the true reachable-set boundary wasn't the
    limiting factor everywhere.
    q_ref: (N,7) -- what to regularize toward (typically broadcast copies of
    the previous waypoint's solution for all K seeds of one env).
    """
    N = q_seed_batch.shape[0]
    q = q_seed_batch.clone()
    if not isinstance(lambda_reg, torch.Tensor):
        lambda_reg = torch.full((N,), float(lambda_reg), device=q.device, dtype=q.dtype)
    lambda_reg = lambda_reg.reshape(N, 1, 1)
    reg_I = (damping) * torch.eye(7, device=q.device, dtype=q.dtype).unsqueeze(0) \
            + lambda_reg * torch.eye(7, device=q.device, dtype=q.dtype).unsqueeze(0)

    for _ in range(iters):
        J, cur_pos, cur_rot = panda_jacobian(q)          # J: (N,6,7)
        pos_err = target_pos - cur_pos
        rot_err = rotmat_to_axis_angle(target_rot @ cur_rot.transpose(-1, -2))
        e = torch.cat([pos_err, rot_err], dim=-1)         # (N,6)

        J_full = J
        if locked_mask.any():
            J_full = J_full.clone()
            J_full[:, :, locked_mask] = 0.0

        JtJ = J_full.transpose(-1, -2) @ J_full            # (N,7,7)
        Jte = (J_full.transpose(-1, -2) @ e.unsqueeze(-1)).squeeze(-1)  # (N,7,6)@(N,6,1) -> (N,7) -- J^T e
        reg_rhs = lambda_reg.reshape(N, 1) * (q_ref - q)   # (N,7)

        A = JtJ + reg_I
        b = Jte + reg_rhs
        dq = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

        q = q + dq
        q = torch.clamp(q, q_lo, q_hi)
        if locked_mask.any():
            q[:, locked_mask] = q_lock[:, locked_mask]

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
    return torch.stack([a1, a2, a3], dim=-2)  # ROWS, matches pytorch3d


def _matrix_to_rot6d(R):
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)


def _project_waypoints_impl(raw_clean, fault_spec, q_prev_seed, K, iters, lambda_reg=0.0):
    orig_device = raw_clean.device
    dtype = torch.float32
    cpu = torch.device("cpu")

    raw_clean_cpu = raw_clean.detach().to(cpu, dtype=dtype)
    B, Tp, Da = raw_clean_cpu.shape

    joint_idx = fault_spec['joint_idx']
    q_lock_raw = fault_spec['q_lock']
    if isinstance(q_lock_raw, torch.Tensor):
        q_lock_per_env = q_lock_raw.detach().to(cpu, dtype=dtype).reshape(B)
    else:
        q_lock_per_env = torch.full((B,), float(q_lock_raw), dtype=dtype)

    q_lo = fault_spec['q_lo'].to(cpu, dtype=dtype)
    q_hi = fault_spec['q_hi'].to(cpu, dtype=dtype)
    is_locked = fault_spec['fault_type'] == 'locked'

    locked_mask = torch.zeros(7, dtype=torch.bool)
    locked_mask[joint_idx] = is_locked

    corrected = raw_clean_cpu.clone()
    q_seed = q_prev_seed.detach().to(cpu, dtype=dtype).clone()  # (7,) or (B,7)
    if q_seed.dim() == 1:
        q_seed = q_seed.unsqueeze(0).expand(B, -1).clone()

    for k in range(Tp):
        pos_k = raw_clean_cpu[:, k, 0:3]
        rot6_k = raw_clean_cpu[:, k, 3:9]
        target_rot_k = _rot6d_to_matrix(rot6_k)

        seeds = torch.rand(B, K, 7, dtype=dtype) * (q_hi - q_lo) + q_lo
        seeds[:, 0, :] = q_seed
        if locked_mask.any():
            seeds[:, :, joint_idx] = q_lock_per_env.unsqueeze(1)

        q_lock_vec_bk = torch.zeros(B, K, 7, dtype=dtype)
        if locked_mask.any():
            q_lock_vec_bk[:, :, joint_idx] = q_lock_per_env.unsqueeze(1)

        # Two-population regularization strategy: pulling ALL K seeds toward
        # the same q_ref collapses multi-start diversity (confirmed
        # empirically -- when the true target is far from q_ref, this made
        # convergence WORSE, not better, since all 64 restarts funnel into
        # the same q_ref-adjacent local minimum instead of spreading out).
        # Instead: half the seeds explore freely (lambda_reg effectively 0
        # via a per-seed lambda vector), half get pulled toward q_ref
        # (continuity-preferring). Final selection (below) already prefers
        # converged+close-to-q_ref, so this preserves both broad search AND
        # continuity preference without sacrificing either.
        half = K // 2
        lambda_per_seed = torch.zeros(K, dtype=dtype)
        lambda_per_seed[half:] = 1.0  # multiplier; actual lambda_reg value applied below
        lambda_per_seed_bk = lambda_per_seed.unsqueeze(0).expand(B, K).clone()

        q_ref_bk = q_seed.unsqueeze(1).expand(B, K, 7).clone()

        N = B * K
        seeds_flat = seeds.reshape(N, 7)
        target_pos_flat = pos_k.unsqueeze(1).expand(B, K, 3).reshape(N, 3)
        target_rot_flat = target_rot_k.unsqueeze(1).expand(B, K, 3, 3).reshape(N, 3, 3)
        q_lock_vec_flat = q_lock_vec_bk.reshape(N, 7)
        q_ref_flat = q_ref_bk.reshape(N, 7)
        lambda_flat = (lambda_per_seed_bk.reshape(N) * lambda_reg)

        q_sol_flat, pos_err_flat, rot_err_flat = solve_batch_ik(
            target_pos_flat, target_rot_flat, seeds_flat, q_lo, q_hi,
            locked_mask, q_lock_vec_flat, q_ref_flat, iters, lambda_reg=lambda_flat)

        q_sol = q_sol_flat.reshape(B, K, 7)
        pos_err = pos_err_flat.reshape(B, K)
        rot_err = rot_err_flat.reshape(B, K)

        converged = (pos_err < 0.01) & (rot_err < 0.15)
        new_seed = torch.zeros(B, 7, dtype=dtype)
        for b in range(B):
            if converged[b].any():
                cand = q_sol[b][converged[b]]
                # NEW selection: among converged, pick the one closest to
                # q_ref (continuity), not just "any converged" -- with the
                # regularized solve this is mostly redundant (solutions
                # already pulled toward q_ref) but breaks ties principled-ly.
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


def project_waypoints(raw_clean, fault_spec, q_prev_seed, K=64, iters=5, lambda_reg=0.5):
    corrected, _ = _project_waypoints_impl(raw_clean, fault_spec, q_prev_seed, K, iters, lambda_reg)
    return corrected


class ProjectWaypoints:
    def __init__(self, K=64, iters=5, lambda_reg=0.0):
        self.K = K
        self.iters = iters
        self.lambda_reg = lambda_reg
        self.last_q_seed = None

    def __call__(self, raw_clean, fault_spec, q_prev_seed):
        corrected, last_seed = _project_waypoints_impl(
            raw_clean, fault_spec, q_prev_seed, self.K, self.iters, self.lambda_reg)
        self.last_q_seed = last_seed
        return corrected


def project_waypoints_to_joint_targets(raw_ee_pose, fault_spec, q_prev_seed, K=64, iters=5, lambda_reg=0.3, q_ref_anchor=None):
    """
    TERMINAL joint-target conversion for actuation_mode='joint'. Deliberately
    NOT sharing code with _project_waypoints_impl (the mid-denoising C-step
    projector) -- kept fully separate per explicit instruction, so a change
    to one can never silently affect the other. Structurally similar (same
    per-waypoint warm-started IK loop) but returns the JOINT solution
    sequence (B, Tp, 7) directly, not an FK-reconstructed EE-pose -- that FK
    round-trip is exactly what actuation_mode='joint' exists to skip.

    raw_ee_pose: (B, Tp, 9) -- pos(3) + rot6d(6), gripper column NOT included
                 (caller keeps gripper separate and reattaches it).
    Returns: q_targets (B, Tp, 7) tensor of joint solutions, one per waypoint,
             on the SAME device/dtype as raw_ee_pose. Also returns the final
             waypoint's solution separately for warm-start chaining across
             predict_action calls (mirrors ProjectWaypoints.last_q_seed).
    """
    orig_device = raw_ee_pose.device
    orig_dtype = raw_ee_pose.dtype
    dtype = torch.float32
    cpu = torch.device("cpu")

    raw_cpu = raw_ee_pose.detach().to(cpu, dtype=dtype)
    B, Tp, _ = raw_cpu.shape

    joint_idx = fault_spec['joint_idx']
    q_lock_raw = fault_spec['q_lock']
    if isinstance(q_lock_raw, torch.Tensor):
        q_lock_per_env = q_lock_raw.detach().to(cpu, dtype=dtype).reshape(B)
    else:
        q_lock_per_env = torch.full((B,), float(q_lock_raw), dtype=dtype)

    q_lo = fault_spec['q_lo'].to(cpu, dtype=dtype)
    q_hi = fault_spec['q_hi'].to(cpu, dtype=dtype)
    is_locked = fault_spec['fault_type'] == 'locked'

    locked_mask = torch.zeros(7, dtype=torch.bool)
    locked_mask[joint_idx] = is_locked

    q_seed = q_prev_seed.detach().to(cpu, dtype=dtype).clone()
    if q_seed.dim() == 1:
        q_seed = q_seed.unsqueeze(0).expand(B, -1).clone()

    if q_ref_anchor is not None:
        q_ref_fixed = q_ref_anchor.detach().to(cpu, dtype=dtype).clone()
        if q_ref_fixed.dim() == 1:
            q_ref_fixed = q_ref_fixed.unsqueeze(0).expand(B, -1).clone()
    else:
        q_ref_fixed = None

    q_targets = torch.zeros(B, Tp, 7, dtype=dtype)

    for k in range(Tp):
        pos_k = raw_cpu[:, k, 0:3]
        rot6d_k = raw_cpu[:, k, 3:9]
        target_rot_k = _rot6d_to_matrix(rot6d_k)

        seeds = torch.rand(B, K, 7, dtype=dtype) * (q_hi - q_lo) + q_lo
        seeds[:, 0, :] = q_seed
        if locked_mask.any():
            seeds[:, :, joint_idx] = q_lock_per_env.unsqueeze(1)

        q_lock_vec_bk = torch.zeros(B, K, 7, dtype=dtype)
        if locked_mask.any():
            q_lock_vec_bk[:, :, joint_idx] = q_lock_per_env.unsqueeze(1)

        anchor = q_ref_fixed if q_ref_fixed is not None else q_seed
        q_ref_bk = anchor.unsqueeze(1).expand(B, K, 7).clone()

        N = B * K
        seeds_flat = seeds.reshape(N, 7)
        target_pos_flat = pos_k.unsqueeze(1).expand(B, K, 3).reshape(N, 3)
        target_rot_flat = target_rot_k.unsqueeze(1).expand(B, K, 3, 3).reshape(N, 3, 3)
        q_lock_vec_flat = q_lock_vec_bk.reshape(N, 7)
        q_ref_flat = q_ref_bk.reshape(N, 7)

        q_sol_flat, pos_err_flat, rot_err_flat = solve_batch_ik(
            target_pos_flat, target_rot_flat, seeds_flat, q_lo, q_hi,
            locked_mask, q_lock_vec_flat, q_ref_flat, iters, lambda_reg=lambda_reg)

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
        q_targets[:, k, :] = q_seed

    return q_targets.to(orig_device, dtype=orig_dtype), q_seed.to(orig_device, dtype=orig_dtype)
