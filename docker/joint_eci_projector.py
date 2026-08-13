"""
Native joint-space E-C-I: implements the PPR (Predict-Project-Renoise)
structure (arXiv 2601.21033) for enforcing actuator-fault constraints
during diffusion denoising, WITHOUT any IK (the policy already outputs
joint values q in R^7 + gripper directly -- this is the whole point of
retraining natively in joint space).

Theoretical grounding (session record): "projecting through the denoiser
keeps samples close to the data manifold... renoising and iterating drive
samples toward the constrained marginal" (PPR, arXiv 2601.21033). Naive
post-hoc or mid-denoising projection WITHOUT a renoise step has no
theoretical guarantee of staying on the policy's learned manifold -- this
module fixes that gap identified this session (previous C-step design only
"blended" the projection back in, which is not equivalent to PPR's renoise
and was flagged as theoretically incomplete).

Three usage modes, sharing this one projector:
  (a) posthoc: project ONLY the final x0 after full unconstrained denoising
      (no manifold-recovery mechanism needed since there's no more
      denoising left to do -- this is the naive baseline for comparison)
  (b) eci: project + renoise at EVERY denoising step (this module's main
      contribution)
  (c) guidance: NOT implemented here -- see embodiment_guidance.py-style
      gradient guidance, kept separate per this project's established
      "guidance is a distinct mechanism from projection" separation.
"""
import torch


def project_fault(q_traj, fault_spec):
    """
    THE Pi_fault operator: closed-form, non-iterative, no IK. Operates on
    a (B, Tp, 7) joint-angle trajectory (gripper dim NOT included -- caller
    slices/concats it separately, matching this project's established
    convention of keeping gripper untouched by fault projection).

    fault_spec: dict with keys:
      joint_idx: int, which joint (0-6) is faulted
      fault_type: None | 'locked' | 'range_reduced' | 'velocity_limited'
      q_lock: (B,) or scalar -- locked value (for 'locked')
      q_lo, q_hi: (7,) -- PER-JOINT feasible range (already narrowed for
                  'range_reduced'; full physical range otherwise)
      v_max: (B,) or scalar -- max |delta q| per denoising-trajectory
             timestep (for 'velocity_limited'); NOTE this is a per-ACTION-
             CHUNK-timestep limit (within the predicted Tp-step trajectory),
             analogous to but distinct from the real-time joint_output_max
             used at the actuation_wrapper layer.
      q_anchor: (B,7) -- starting point for the velocity-limited recursion
                (the actual current robot qpos, NOT the previous denoising
                iterate's q_anchor -- mirrors this session's earlier
                q_ref_anchor fix: the physical starting point must be the
                real robot state, fixed for the whole episode/chunk, not
                something that drifts step to step).
    Returns: (B, Tp, 7) projected joint trajectory.
    """
    if fault_spec is None or fault_spec.get('fault_type') is None:
        return q_traj

    B, Tp, _ = q_traj.shape
    j = fault_spec['joint_idx']
    q_lo = fault_spec['q_lo'].to(q_traj.device, q_traj.dtype)
    q_hi = fault_spec['q_hi'].to(q_traj.device, q_traj.dtype)
    out = q_traj.clone()

    if fault_spec['fault_type'] == 'locked':
        q_lock = fault_spec['q_lock']
        if not torch.is_tensor(q_lock):
            q_lock = torch.full((B,), float(q_lock), device=q_traj.device, dtype=q_traj.dtype)
        else:
            q_lock = q_lock.to(q_traj.device, q_traj.dtype)
        out[:, :, j] = q_lock.unsqueeze(1).expand(B, Tp)

    elif fault_spec['fault_type'] == 'range_reduced':
        out[:, :, j] = torch.clamp(out[:, :, j], q_lo[j], q_hi[j])

    elif fault_spec['fault_type'] == 'velocity_limited':
        v_max = fault_spec['v_max']
        if not torch.is_tensor(v_max):
            v_max = torch.full((B,), float(v_max), device=q_traj.device, dtype=q_traj.dtype)
        else:
            v_max = v_max.to(q_traj.device, q_traj.dtype)
        q_anchor = fault_spec['q_anchor'].to(q_traj.device, q_traj.dtype)  # (B,7)
        prev = q_anchor[:, j]  # (B,)
        limited = torch.zeros(B, Tp, device=q_traj.device, dtype=q_traj.dtype)
        for t in range(Tp):
            delta = torch.clamp(out[:, t, j] - prev, -v_max, v_max)
            cur = prev + delta
            limited[:, t] = cur
            prev = cur
        out[:, :, j] = limited
    else:
        raise ValueError(f"unknown fault_type: {fault_spec['fault_type']}")

    # ALWAYS clip to the (possibly narrowed) hard physical range, for every
    # joint, regardless of fault type -- healthy joints too, matching
    # standard safety-limit practice and guaranteeing the output is always
    # kinematically valid even if upstream logic has a bug.
    out = torch.clamp(out, q_lo.view(1, 1, 7), q_hi.view(1, 1, 7))
    return out


def eci_conditional_sample(policy, condition_data, condition_mask, fault_spec,
        local_cond=None, global_cond=None, generator=None, **kwargs):
    """
    PPR-structured E-C-I denoising loop for actuation_mode='joint' native
    checkpoints. Drop-in replacement for
    DiffusionUnetHybridImagePolicy.conditional_sample, active ONLY when
    fault_spec is not None (mirrors this project's established gating
    convention -- see favor_policy.py's `if self.fault_spec is not None`).

    Implements DDPMScheduler's step formula manually (not calling
    scheduler.step()) because we need direct access to the intermediate
    x0 (Tweedie) estimate to project it BEFORE the posterior/renoise step --
    diffusers' scheduler.step() does not expose this intermediate value.
    """
    model = policy.model
    scheduler = policy.noise_scheduler
    trajectory = torch.randn(
        size=condition_data.shape, dtype=condition_data.dtype,
        device=condition_data.device, generator=generator)

    scheduler.set_timesteps(policy.num_inference_steps)
    alphas_cumprod = scheduler.alphas_cumprod.to(trajectory.device)

    action_dim = 8  # joint(7) + gripper(1), this project's native joint-space convention
    q_slice = slice(0, 7)

    for t in scheduler.timesteps:
        trajectory[condition_mask] = condition_data[condition_mask]
        model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

        # ---- E: Tweedie x0 estimate (manual, matches DDPMScheduler's epsilon-pred formula) ----
        alpha_bar_t = alphas_cumprod[t]
        sqrt_alpha_bar_t = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
        x0_hat = (trajectory - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
        if scheduler.config.clip_sample:
            x0_hat = x0_hat.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

        # ---- C: project the joint-angle dims only; gripper untouched ----
        x0_proj = x0_hat.clone()
        x0_proj[..., q_slice] = project_fault(x0_hat[..., q_slice], fault_spec)

        # ---- I: renoise from the PROJECTED x0 via the unconstrained forward
        # kernel to the NEXT timestep t-1 (PPR's "renoise" -- this is what
        # restores manifold consistency; a plain blend/clip here would not).
        t_idx = (scheduler.timesteps == t).nonzero()[0].item()
        if t_idx + 1 < len(scheduler.timesteps):
            t_prev = scheduler.timesteps[t_idx + 1]
            alpha_bar_prev = alphas_cumprod[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=trajectory.device)  # t=0 endpoint: no more noise

        noise = torch.randn(trajectory.shape, device=trajectory.device,
                             dtype=trajectory.dtype, generator=generator)
        trajectory = alpha_bar_prev.sqrt() * x0_proj + (1 - alpha_bar_prev).sqrt() * noise

    trajectory[condition_mask] = condition_data[condition_mask]
    # Final hard-enforcement: guarantee feasibility even at t=0 (the renoise
    # loop's last iterate is already alpha_bar_prev=1 -> deterministic x0_proj,
    # but re-apply projection once more for numerical safety / clarity).
    trajectory_final = trajectory.clone()
    trajectory_final[..., q_slice] = project_fault(trajectory[..., q_slice], fault_spec)
    return trajectory_final
