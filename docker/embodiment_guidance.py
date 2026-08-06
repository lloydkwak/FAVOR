"""
Embodiment-Aware Diffusion Guidance (EADP), adapted from UMI-on-Air
(Gupta et al., "UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic
Visuomotor Policies", ICRA 2026, arXiv 2510.02614), Section III-B/C, Eq.(1)(2)(8),
Algorithm 1.

DELIBERATELY SEPARATE from ik_projector.py's C-step (both the mid-denoising
_project_waypoints_impl and the terminal project_waypoints_to_joint_targets):
no shared state, no shared code path. Only pure, already-unit-tested math
primitives (panda_fk, rotmat_to_axis_angle) are imported, since those are
verified building blocks rather than "C-step logic".

What THIS module adds beyond the original paper (paper has no such term):
the fault-aware f_IK -- solve_batch_ik-style locked/range-reduced joint
handling -- is folded into differentiable_ik_step() below, taking the place
of the paper's plain (fault-unaware) f_IK in Eq.(1).

Original paper's f_IK is a single deterministic (no multi-restart) warm-started
IK step -- NOT the K=64 random-restart multi-modal search used in the C-step /
terminal joint conversion. This is intentional: guidance needs a SMOOTH,
differentiable cost, and argmin-over-K-seeds is not differentiable in a useful
way for gradient guidance.
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from panda_kinematics import panda_fk, panda_jacobian
from ik_projector import rotmat_to_axis_angle  # pure math primitive, already unit-tested


def _rot6d_to_matrix_diff(rot6):
    """Differentiable rotation_6d -> matrix (row-based, matches pytorch3d
    convention -- same formula as ik_projector._rot6d_to_matrix, duplicated
    here on purpose to keep this module free of any C-step import beyond
    pure math primitives)."""
    a1, a2 = rot6[..., 0:3], rot6[..., 3:6]
    a1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a2 = a2 - (a1 * a2).sum(dim=-1, keepdim=True) * a1
    a2 = a2 / a2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-2)


def differentiable_ik_step(q, target_pos, target_rot, fault_spec, iters=3, damping=1e-3):
    """
    Single warm-started Newton-Gauss IK solve, DIFFERENTIABLE end-to-end
    w.r.t. target_pos/target_rot. No K-restart search (deterministic, smooth).
    q: (B,7) current joint config (the warm start / current robot state).
    fault_spec: same dict shape as elsewhere ('joint_idx','q_lock','fault_type','q_lo','q_hi'),
                q_lock may be a (B,) tensor (per-env) or python scalar.
    Returns q_ik: (B,7), differentiable.
    """
    B = q.shape[0]
    device, dtype = q.device, q.dtype
    q_lo = fault_spec['q_lo'].to(device=device, dtype=dtype)
    q_hi = fault_spec['q_hi'].to(device=device, dtype=dtype)
    joint_idx = fault_spec['joint_idx']
    is_locked = fault_spec['fault_type'] == 'locked'
    locked_mask = torch.zeros(7, dtype=torch.bool, device=device)
    locked_mask[joint_idx] = is_locked

    q_lock_raw = fault_spec['q_lock']
    if isinstance(q_lock_raw, torch.Tensor):
        q_lock_vec = torch.zeros(B, 7, device=device, dtype=dtype)
        q_lock_vec[:, joint_idx] = q_lock_raw.to(device=device, dtype=dtype).reshape(B)
    else:
        q_lock_vec = torch.zeros(B, 7, device=device, dtype=dtype)
        if is_locked:
            q_lock_vec[:, joint_idx] = float(q_lock_raw)

    q_cur = q
    for _ in range(iters):
        J, cur_pos, cur_rot = panda_jacobian(q_cur)
        pos_err = target_pos - cur_pos
        rot_err = rotmat_to_axis_angle(target_rot @ cur_rot.transpose(-1, -2))
        e = torch.cat([pos_err, rot_err], dim=-1)

        J_full = J
        if locked_mask.any():
            J_full = J_full.clone()
            J_full[:, :, locked_mask] = 0.0

        JtJ = J_full.transpose(-1, -2) @ J_full
        Jte = (J_full.transpose(-1, -2) @ e.unsqueeze(-1)).squeeze(-1)
        reg_I = damping * torch.eye(7, device=device, dtype=dtype).unsqueeze(0)
        dq = torch.linalg.solve(JtJ + reg_I, Jte.unsqueeze(-1)).squeeze(-1)

        q_cur = q_cur + dq
        q_cur = torch.clamp(q_cur, q_lo, q_hi)
        if locked_mask.any():
            # Differentiable-safe overwrite: locked columns get a constant
            # (q_lock, independent of target_pos/target_rot), so autograd
            # correctly assigns them zero gradient w.r.t. the trajectory --
            # this is the mathematically correct behavior (the locked joint
            # truly cannot respond to changes in the target), not a bug.
            mask_f = locked_mask.to(dtype)
            q_cur = q_cur * (1 - mask_f) + q_lock_vec * mask_f
    return q_cur


def rollout_tracking_cost(a_ee_raw, fault_spec, q_current, delta_max_vec, ik_iters=3):
    """
    Implements UMI-on-Air Eq.(1)(2), fault-aware:
        q_{t+1} = q_t + clip(f_IK^fault(a_t, q_t) - q_t, -delta_max, delta_max)
        L_track(a) = sum_t || f_FK(q_t) - a_t ||^2   (position + orientation)

    a_ee_raw: (B, Tp, 9) pos(3)+rot6d(6), UNNORMALIZED (real units), the
              CURRENT noisy trajectory sample a^k (per Algorithm 1 -- this is
              applied to the noisy sample directly, not a Tweedie/clean
              estimate, matching the paper exactly).
    q_current: (B,7) actual robot joint state (warm start for the whole
               waypoint-by-waypoint rollout, i.e. episode-persistent state,
               same role as _q_prev_seed elsewhere but a SEPARATE variable).
    delta_max_vec: (7,) or (B,7) per-joint max step -- this is where
                   velocity_limited faults are represented (see design note
                   in the earlier conversation: this parameter, not the
                   EE-pose IK constraint, is where velocity_limited belongs).
    Returns: L_track, a (B,) tensor, differentiable w.r.t. a_ee_raw.
    """
    B, Tp, _ = a_ee_raw.shape
    q = q_current
    total = torch.zeros(B, device=a_ee_raw.device, dtype=a_ee_raw.dtype)
    for k in range(Tp):
        pos_target = a_ee_raw[:, k, 0:3]
        rot_target = _rot6d_to_matrix_diff(a_ee_raw[:, k, 3:9])

        q_ik = differentiable_ik_step(q, pos_target, rot_target, fault_spec, iters=ik_iters)
        delta = torch.clamp(q_ik - q, -delta_max_vec, delta_max_vec)
        q_next = q + delta

        fk_pos, fk_rot = panda_fk(q_next)
        pos_sq_err = (fk_pos - pos_target).pow(2).sum(-1)
        rot_axis_angle = rotmat_to_axis_angle(rot_target @ fk_rot.transpose(-1, -2))
        rot_sq_err = rot_axis_angle.pow(2).sum(-1)

        total = total + pos_sq_err + rot_sq_err
        q = q_next
    return total, q  # also return final q for warm-start chaining, like elsewhere


class EmbodimentGuidance:
    """
    Stateful wrapper: holds the episode-persistent robot-state warm start
    (q_current) for the tracking-cost rollout, separate from any C-step or
    terminal-IK state elsewhere.
    """
    def __init__(self, delta_max=0.2, ik_iters=3, lambda_scale=0.1):
        # lambda_scale=0.1 calibrated empirically (test_guidance_magnitude_calibration.py):
        # nudge magnitude ~= 5% of typical normalized-trajectory element scale
        # per denoising step -- large enough to have a real effect over ~100
        # steps, far below the runaway magnitude seen at lambda_scale=50
        # (476.9 total |diff| across 40 elements, i.e. mean ~11.9/element,
        # completely overwhelming the trajectory).
        self.delta_max = delta_max
        self.ik_iters = ik_iters
        self.lambda_scale = lambda_scale
        self.q_current = None  # set externally each predict_action call (live robot qpos)

    def guide(self, trajectory_norm, normalizer, fault_spec, alpha_prod_t):
        """
        Implements Algorithm 1, line 3: ã^k = a^k - lambda * omega_k * grad_{a^k} L_track(a^k)
        trajectory_norm: (B, Tp, Da) NORMALIZED trajectory (a^k), as used
                          internally by conditional_sample -- gradient flows
                          through the (differentiable, affine) unnormalize.
        Returns: nudged trajectory_norm (same shape), detached from the
                 guidance graph (so it doesn't accumulate across denoising
                 steps -- matches the paper's per-step independent guidance).
        """
        if self.q_current is None:
            raise RuntimeError("EmbodimentGuidance.q_current must be set (live robot qpos) before guide() is called")
        B = trajectory_norm.shape[0]
        device, dtype = trajectory_norm.device, trajectory_norm.dtype
        delta_max_vec = torch.full((7,), self.delta_max, device=device, dtype=dtype)

        traj_for_grad = trajectory_norm.detach().clone().requires_grad_(True)
        raw = normalizer["action"].unnormalize(traj_for_grad)  # (B,Tp,Da), Da=10 (pos+rot6d+gripper)
        raw_ee = raw[..., 0:9]  # guidance only touches pos+rot6d, not gripper

        L_track, final_q = rollout_tracking_cost(
            raw_ee, fault_spec, self.q_current, delta_max_vec, ik_iters=self.ik_iters)
        loss = L_track.sum()
        grad, = torch.autograd.grad(loss, traj_for_grad)

        omega_k = alpha_prod_t  # bar_alpha_k, matches paper's guidance schedule exactly
        nudged = trajectory_norm - self.lambda_scale * omega_k * grad
        self.q_current = final_q.detach()
        return nudged.detach()
