"""
Native joint-space E-C-I: implements the PPR (Predict-Project-Renoise)
structure (arXiv 2601.21033) for enforcing actuator-fault constraints
during diffusion denoising, WITHOUT any IK.

REDESIGNED this session: project_fault now applies to ALL 7 joints
uniformly, not just a single "known faulted joint". Rationale: which
joint (if any) is faulted is not something the projection mechanism
should need to know about specially -- in deployment, any joint could be
the one that's actually faulted, so the constraint is applied per-joint
across the board, using each joint's own valid range. A healthy joint's
range is simply its full physical range, making the projection an
identity operation for that joint (verified: this is why B1 and FAVOR
matched exactly under a vacuous/full-range test this session, once the
earlier normalized-vs-physical-units clamp bug was also fixed). A faulted
joint's range is narrowed (locked = zero-width range at q_lock,
range_reduced = narrowed range, velocity_limited = per-step delta cap).
This also means the mechanism no longer assumes exactly one joint is
faulted -- it would behave correctly (each joint independently) even
under multiple simultaneous joint faults, though this project's current
experiments still test one faulted joint at a time.
"""
import torch


def project_fault(q_traj, q_lo, q_hi, v_max=None, q_anchor=None):
    """
    THE Pi_fault operator, applied to ALL 7 joints simultaneously.

    CRITICAL UNIT REQUIREMENT (bug found and fixed this session): q_traj,
    q_lo, q_hi, v_max, q_anchor must ALL be in the SAME space as q_traj
    actually lives in at the call site. Inside eci_conditional_sample,
    q_traj (x0_hat's joint slice) is in the diffusion model's NORMALIZED
    space (confirmed empirically: roughly [-0.76, 1.00], NOT physical
    radians [-2.9, 2.9]), because normalization is a PER-JOINT AFFINE
    transform (different scale AND offset per joint via LinearNormalizer),
    not a simple global rescaling. Passing physical-radian q_lo/q_hi
    directly caused every joint's value to be clipped against numerically
    wrong bounds on every denoising step, catastrophically degrading
    output even under a "vacuous" (physically full-range) fault_spec
    (confirmed: Lift collapsed from 0.92 to 0.04). Callers MUST normalize
    q_lo/q_hi (and q_anchor) into this space before calling project_fault
    from within eci_conditional_sample -- see the norm_bounds() helper
    below, used by NativeJointPolicy to do this conversion once per
    predict_action call (not per denoising step).

    q_traj: (B, Tp, 7) trajectory in whatever space it's given in.
    q_lo, q_hi: (7,) per-joint valid range, in the SAME space as q_traj.
    v_max: (7,) or None. Per-joint max |delta| per timestep, same space.
    q_anchor: (B,7) or None. Starting point for velocity recursion, same space.
    Returns: (B, Tp, 7) projected trajectory, same space as input.
    """
    B, Tp, _ = q_traj.shape
    device, dtype = q_traj.device, q_traj.dtype
    out = q_traj.clone()

    def _bshape(t, trailing_ones):
        # Accepts (7,) or (B,7); returns a tensor broadcastable against
        # (B, *trailing_ones, 7). Bug fix (this session): the dynamic
        # per-episode fault_spec built from the environment has (B,7)
        # bounds (each env instance's actual fault value can differ),
        # while earlier static tests used (7,) -- this makes project_fault
        # agnostic to which shape it receives.
        t = t.to(device, dtype)
        if t.dim() == 1:
            return t.view(1, *([1] * trailing_ones), 7)
        else:
            return t.view(B, *([1] * trailing_ones), 7)

    if v_max is not None:
        assert q_anchor is not None, "q_anchor required when v_max is given"
        v_max_b = _bshape(v_max, 0)
        q_anchor = q_anchor.to(device, dtype)  # (B,7)
        prev = q_anchor  # (B,7)
        limited = torch.zeros(B, Tp, 7, device=device, dtype=dtype)
        for t in range(Tp):
            delta = torch.clamp(out[:, t, :] - prev, -v_max_b, v_max_b)
            cur = prev + delta
            limited[:, t, :] = cur
            prev = cur
        out = limited

    # Hard range clip, every joint, always. For healthy joints this is a
    # numerically exact identity operation (q_lo/q_hi cover the joint's
    # full physical range, which the policy was trained within). For
    # faulted joints (locked: q_lo==q_hi; range_reduced: narrowed) this
    # enforces the actual constraint.
    q_lo_b = _bshape(q_lo, 1)
    q_hi_b = _bshape(q_hi, 1)
    out = torch.clamp(out, q_lo_b, q_hi_b)
    return out


def eci_conditional_sample(policy, condition_data, condition_mask, fault_spec,
        local_cond=None, global_cond=None, generator=None, n_resample=1, **kwargs):
    """
    n_resample: number of Predict-Project-Renoise iterations PER NOISE LEVEL
    before advancing to the next (lower) level, following PPR's corrector-kernel
    formulation and RePaint's resampling trick (jump down one step via renoise,
    then back up one step via the forward kernel, repeated) rather than a
    single pass. Session finding motivating this: independently clipping one
    joint dimension does not by itself respect the network's LEARNED joint
    correlations across all 7 dims (the manifold is not axis-aligned), so a
    single renoise pass may not give the network enough cycles to adapt the
    other 6 joints to be consistent with the fixed joint. n_resample=1
    reproduces the exact prior single-pass behavior (default, backward
    compatible). Deliberately NOT using classical Jacobian-based nullspace
    redistribution here -- that would reintroduce exactly the kind of
    externally-imposed, network-never-trained-on configuration that native
    joint-space retraining was meant to eliminate (the EE+IK OOD problem from
    earlier this session). This iterates the network's OWN learned corrector
    instead.
    """
    """
    PPR-structured E-C-I denoising loop for actuation_mode='joint' native
    checkpoints. fault_spec is now a dict with keys q_lo, q_hi, v_max
    (optional), q_anchor (required if v_max given) -- see project_fault.
    fault_spec=None means "fully unconstrained" (equivalent to calling the
    original conditional_sample); provided mainly for API symmetry, callers
    typically pass a fault_spec with all-healthy (full-range) joints instead
    to exercise this code path identically to a real fault scenario minus
    the actual narrowing.
    """
    model = policy.model
    scheduler = policy.noise_scheduler
    trajectory = torch.randn(
        size=condition_data.shape, dtype=condition_data.dtype,
        device=condition_data.device, generator=generator)

    scheduler.set_timesteps(policy.num_inference_steps)
    alphas_cumprod = scheduler.alphas_cumprod.to(trajectory.device)
    q_slice = slice(0, 7)

    alphas = scheduler.alphas.to(trajectory.device)  # single-step (non-cumulative)
    # signal-retention, confirmed present on DDPMScheduler from its own
    # step() source (self.alphas[t] used directly there) -- needed for the
    # RePaint-style "jump back up one step" renoise below.

    for t in scheduler.timesteps:
        t_idx = (scheduler.timesteps == t).nonzero()[0].item()
        if t_idx + 1 < len(scheduler.timesteps):
            t_prev = scheduler.timesteps[t_idx + 1]
            alpha_bar_prev = alphas_cumprod[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=trajectory.device)

        for resample_i in range(n_resample):
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # ---- E: Tweedie x0 estimate ----
            alpha_bar_t = alphas_cumprod[t]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
            x0_hat = (trajectory - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
            if scheduler.config.clip_sample:
                x0_hat = x0_hat.clamp(-1.0, 1.0)

            # ---- C: project ALL joints (identity for healthy ones) ----
            x0_proj = x0_hat.clone()
            if fault_spec is not None:
                x0_proj[..., q_slice] = project_fault(
                    x0_hat[..., q_slice], fault_spec['q_lo'], fault_spec['q_hi'],
                    v_max=fault_spec.get('v_max'), q_anchor=fault_spec.get('q_anchor'))

            # ---- I: renoise from the PROJECTED x0 via the unconstrained
            # forward kernel to the next timestep t-1 (PPR's "renoise") ----
            noise = torch.randn(trajectory.shape, device=trajectory.device,
                                 dtype=trajectory.dtype, generator=generator)
            x_prev = alpha_bar_prev.sqrt() * x0_proj + (1 - alpha_bar_prev).sqrt() * noise

            is_last_resample = (resample_i == n_resample - 1)
            is_last_timestep = (t_idx + 1 >= len(scheduler.timesteps))
            if is_last_resample or is_last_timestep:
                # commit: actually advance to t_prev
                trajectory = x_prev
            else:
                # RePaint-style resampling: jump BACK UP one step from x_prev
                # via the single-step forward kernel q(x_t | x_{t-1}), so the
                # network re-predicts at the SAME noise level t with a fresh
                # sample that already reflects the projected (constrained)
                # x0 from this pass -- giving it another cycle to reconcile
                # the other 6 joints with the fixed one before moving on.
                alpha_t_single = alphas[t]
                noise_up = torch.randn(trajectory.shape, device=trajectory.device,
                                        dtype=trajectory.dtype, generator=generator)
                trajectory = alpha_t_single.sqrt() * x_prev + (1 - alpha_t_single).sqrt() * noise_up

    trajectory[condition_mask] = condition_data[condition_mask]
    trajectory_final = trajectory.clone()
    if fault_spec is not None:
        trajectory_final[..., q_slice] = project_fault(
            trajectory[..., q_slice], fault_spec['q_lo'], fault_spec['q_hi'],
            v_max=fault_spec.get('v_max'), q_anchor=fault_spec.get('q_anchor'))
    return trajectory_final

def normalize_joint_bounds(policy, q_lo_phys, q_hi_phys, v_max_phys=None, q_anchor_phys=None):
    """
    Converts physical-radian joint bounds into the policy's normalized
    action space, using policy.normalizer['action'] (the SAME normalizer
    applied everywhere else in this pipeline). Call ONCE per predict_action
    (not per denoising step) and pass the result into eci_conditional_sample.

    The action normalizer operates on the full 8-dim action (7 joints +
    gripper); we build dummy 8-dim vectors (gripper=0, irrelevant) purely
    to reuse the exact same per-joint affine transform the policy was
    trained with, then discard the gripper dim.
    """
    device = policy.device
    dtype = policy.dtype
    normalizer = policy.normalizer['action']

    def _norm(q_phys):
        # Accepts either (7,) or (B,7). BUG FIX (found this session): the
        # dynamic per-episode fault_spec built from the environment has
        # (B,7) q_lo/q_hi (each env instance's actual lock/range value can
        # differ), whereas the static vacuous-test fault_spec used (7,) --
        # this function must handle both.
        q_phys = q_phys.to(device, dtype)
        was_1d = (q_phys.dim() == 1)
        if was_1d:
            q_phys = q_phys.unsqueeze(0)  # (1,7)
        B = q_phys.shape[0]
        dummy = torch.zeros(B, 8, device=device, dtype=dtype)
        dummy[:, :7] = q_phys
        out = normalizer.normalize(dummy)[:, :7]
        return out.squeeze(0) if was_1d else out

    q_lo_norm = _norm(q_lo_phys)
    q_hi_norm = _norm(q_hi_phys)
    # NOTE: normalization can flip lo/hi ordering per-joint if the affine
    # transform has a negative scale for that joint -- re-sort elementwise
    # to guarantee lo <= hi after transform, matching torch.clamp's
    # requirement.
    lo_final = torch.minimum(q_lo_norm, q_hi_norm)
    hi_final = torch.maximum(q_lo_norm, q_hi_norm)

    v_max_norm = None
    if v_max_phys is not None:
        # v_max is a DELTA (difference of two values), not an absolute
        # position -- normalize by SCALE only, not offset (confirmed via
        # source: _normalize does x*scale+offset, so a delta transforms as
        # delta*scale, with no offset term). scale/offset confirmed to live
        # at normalizer.params_dict['scale']/['offset'] (dict with 8 entries
        # for the 8-dim action; slice to the first 7 = joints).
        scale = normalizer.params_dict['scale'][:7].to(device, dtype)
        v_max_norm = (v_max_phys.to(device, dtype) * scale).abs()

    q_anchor_norm = None
    if q_anchor_phys is not None:
        B = q_anchor_phys.shape[0]
        dummy = torch.zeros(B, 8, device=device, dtype=dtype)
        dummy[:, :7] = q_anchor_phys.to(device, dtype)
        q_anchor_norm = normalizer.normalize(dummy)[:, :7]

    return lo_final, hi_final, v_max_norm, q_anchor_norm
