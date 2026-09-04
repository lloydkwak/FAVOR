"""
Native joint-space E-C-I (RoboTwin / DP port), 14-dim dual-arm action space.

Ported from the robosuite/Panda version (docker/joint_eci_projector.py).
Same PPR (Predict-Project-Renoise, arXiv 2601.21033) structure and the
same "project ALL joints uniformly, healthy joints get an identity
projection" design -- see that file's docstring for the full rationale,
which is unchanged here.

WHAT IS DIFFERENT FROM THE PANDA VERSION, AND WHY:

  Action layout. DP's zarr action/state vectors are 14-dim, packed as
  [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
  (confirmed from XPolicyLab.utils.process_data.pack_robot_state's
  docstring: dual-arm order is [arm_0, ee_0, arm_1, ee_1], and
  _get_state_keys confirms arm_0/ee_0 = left, arm_1/ee_1 = right). Index
  map:
      0-5:  fl_joint1..6   (left arm)
      6:    left gripper
      7-12: fr_joint1..6   (right arm)
      13:   right gripper

  The Panda version's q_slice = slice(0, 7) was contiguous because its
  8-dim action was [7 joints, 1 gripper] with the gripper last. Here the
  12 arm joints are NOT contiguous (a gripper sits at index 6, splitting
  them), so ARM_IDX below is an explicit index tensor instead of a slice,
  and every place the Panda version used [..., q_slice] this version
  uses [..., ARM_IDX] (fancy indexing; PyTorch supports assignment
  through it the same as it does through a slice).

  Per-joint bounds are 12-wide, not 7-wide, in the same
  [left_arm(6), right_arm(6)] order as ARM_IDX -- i.e. NOT the 14-dim
  action order (no gripper slots in q_lo/q_hi/v_max/q_anchor).

  Everything else (Tweedie x0 estimate, project_fault's clamp+velocity
  logic, PPR renoise, n_resample, normalize_joint_bounds' physical <->
  normalized conversion via the policy's LinearNormalizer) is structurally
  identical to the Panda version.
"""
import torch

# Indices of the 12 arm joints within the 14-dim action vector, in
# [left_arm(6), right_arm(6)] order -- i.e. ARM_IDX[0:6] are the left arm
# joints, ARM_IDX[6:12] are the right arm joints. Index 6 (left gripper)
# and index 13 (right gripper) are excluded.
ARM_IDX = torch.tensor([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=torch.long)
NUM_ARM_JOINTS = 12  # len(ARM_IDX)


def project_fault(q_traj, q_lo, q_hi, v_max=None, q_anchor=None):
    """
    THE Pi_fault operator, applied to all 12 arm joints simultaneously
    (grippers are never passed through this function -- callers slice
    them out via ARM_IDX before calling, same convention as the Panda
    version's 7-joint slice).

    CRITICAL UNIT REQUIREMENT: q_traj, q_lo, q_hi, v_max, q_anchor must
    ALL be in the SAME space as q_traj actually lives in at the call
    site -- inside eci_conditional_sample that is the diffusion model's
    NORMALIZED space, NOT physical radians. Use normalize_joint_bounds()
    to convert once per predict_action call before calling this from
    within the denoising loop.

    q_traj: (B, Tp, 12) trajectory (already sliced to ARM_IDX), whatever
        space it's given in.
    q_lo, q_hi: (12,) or (B, 12) per-joint valid range, SAME space as q_traj,
        in [left_arm(6), right_arm(6)] order.
    v_max: (12,) or (B, 12) or None. Per-joint max |delta| per timestep.
    q_anchor: (B, 12) or None. Starting point for velocity recursion.
    Returns: (B, Tp, 12) projected trajectory, same space as input.
    """
    B, Tp, _ = q_traj.shape
    device, dtype = q_traj.device, q_traj.dtype
    out = q_traj.clone()

    def _bshape(t, trailing_ones):
        t = t.to(device, dtype)
        if t.dim() == 1:
            return t.view(1, *([1] * trailing_ones), NUM_ARM_JOINTS)
        else:
            return t.view(B, *([1] * trailing_ones), NUM_ARM_JOINTS)

    if v_max is not None:
        assert q_anchor is not None, "q_anchor required when v_max is given"
        v_max_b = _bshape(v_max, 0)
        q_anchor = q_anchor.to(device, dtype)  # (B,12)
        prev = q_anchor
        limited = torch.zeros(B, Tp, NUM_ARM_JOINTS, device=device, dtype=dtype)
        for t in range(Tp):
            delta = torch.clamp(out[:, t, :] - prev, -v_max_b, v_max_b)
            cur = prev + delta
            limited[:, t, :] = cur
            prev = cur
        out = limited

    q_lo_b = _bshape(q_lo, 1)
    q_hi_b = _bshape(q_hi, 1)
    out = torch.clamp(out, q_lo_b, q_hi_b)
    return out


def posthoc_conditional_sample(policy, condition_data, condition_mask, fault_spec,
        local_cond=None, global_cond=None, generator=None, **kwargs):
    """
    Baseline projection mode: run the ORIGINAL, completely unconstrained
    policy.conditional_sample() to completion, then apply project_fault
    exactly ONCE to the final trajectory. No per-step intervention, no PPR
    renoise -- this is the "B1 + single post-hoc clip" condition, and is
    the correct identity-check partner for B1 (NOT eci_conditional_sample;
    see module-level note below on why eci is not expected to match B1
    bit-for-bit even under a vacuous fault_spec).
    """
    trajectory = policy.conditional_sample(
        condition_data, condition_mask, local_cond=local_cond,
        global_cond=global_cond, generator=generator, **kwargs,
    )
    arm_idx = ARM_IDX.to(trajectory.device)
    trajectory_final = trajectory.clone()
    if fault_spec is not None:
        trajectory_final[..., arm_idx] = project_fault(
            trajectory[..., arm_idx], fault_spec["q_lo"], fault_spec["q_hi"],
            v_max=fault_spec.get("v_max"), q_anchor=fault_spec.get("q_anchor"))
    return trajectory_final


def eci_conditional_sample(policy, condition_data, condition_mask, fault_spec,
        local_cond=None, global_cond=None, generator=None, n_resample=1, **kwargs):
    """
    PPR-structured E-C-I denoising loop for the RoboTwin/DP 14-dim joint
    action space. fault_spec is a dict with keys q_lo, q_hi (each (12,) or
    (B,12), in ARM_IDX / [left_arm(6), right_arm(6)] order), optionally
    v_max and q_anchor -- see project_fault. fault_spec=None reproduces
    the original unconstrained conditional_sample exactly.

    Structurally identical to the Panda version's eci_conditional_sample
    (same Tweedie/project/renoise/n_resample loop against
    policy.noise_scheduler, a diffusers DDPMScheduler) -- only the joint
    slicing (ARM_IDX instead of slice(0,7)) and bound width (12 vs 7)
    differ.
    """
    model = policy.model
    scheduler = policy.noise_scheduler
    trajectory = torch.randn(
        size=condition_data.shape, dtype=condition_data.dtype,
        device=condition_data.device, generator=generator)
    scheduler.set_timesteps(policy.num_inference_steps)
    alphas_cumprod = scheduler.alphas_cumprod.to(trajectory.device)
    arm_idx = ARM_IDX.to(trajectory.device)
    alphas = scheduler.alphas.to(trajectory.device)

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

            alpha_bar_t = alphas_cumprod[t]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
            x0_hat = (trajectory - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
            if scheduler.config.clip_sample:
                x0_hat = x0_hat.clamp(-1.0, 1.0)

            x0_proj = x0_hat.clone()
            if fault_spec is not None:
                x0_proj[..., arm_idx] = project_fault(
                    x0_hat[..., arm_idx], fault_spec["q_lo"], fault_spec["q_hi"],
                    v_max=fault_spec.get("v_max"), q_anchor=fault_spec.get("q_anchor"))

            noise = torch.randn(trajectory.shape, device=trajectory.device,
                                 dtype=trajectory.dtype, generator=generator)
            x_prev = alpha_bar_prev.sqrt() * x0_proj + (1 - alpha_bar_prev).sqrt() * noise

            is_last_resample = (resample_i == n_resample - 1)
            is_last_timestep = (t_idx + 1 >= len(scheduler.timesteps))
            if is_last_resample or is_last_timestep:
                trajectory = x_prev
            else:
                alpha_t_single = alphas[t]
                noise_up = torch.randn(trajectory.shape, device=trajectory.device,
                                        dtype=trajectory.dtype, generator=generator)
                trajectory = alpha_t_single.sqrt() * x_prev + (1 - alpha_t_single).sqrt() * noise_up

    trajectory[condition_mask] = condition_data[condition_mask]
    trajectory_final = trajectory.clone()
    if fault_spec is not None:
        trajectory_final[..., arm_idx] = project_fault(
            trajectory[..., arm_idx], fault_spec["q_lo"], fault_spec["q_hi"],
            v_max=fault_spec.get("v_max"), q_anchor=fault_spec.get("q_anchor"))
    return trajectory_final


def normalize_joint_bounds(policy, q_lo_phys, q_hi_phys, v_max_phys=None, q_anchor_phys=None):
    """
    Converts physical-radian joint bounds (12-wide, [left_arm(6),
    right_arm(6)] order) into the policy's normalized action space, using
    policy.normalizer['action'] -- the same LinearNormalizer applied
    everywhere else in the DP pipeline.

    Confirmed on a real DP checkpoint: normalizer.params_dict has
    'offset'/'scale' keys, each shape (14,).
    """
    device = policy.device
    dtype = policy.dtype
    normalizer = policy.normalizer["action"]

    def _norm(q_phys):
        q_phys = q_phys.to(device, dtype)
        was_1d = (q_phys.dim() == 1)
        if was_1d:
            q_phys = q_phys.unsqueeze(0)  # (1,12)
        B = q_phys.shape[0]
        dummy = torch.zeros(B, 14, device=device, dtype=dtype)
        dummy[:, ARM_IDX] = q_phys
        out = normalizer.normalize(dummy)[:, ARM_IDX]
        return out.squeeze(0) if was_1d else out

    q_lo_norm = _norm(q_lo_phys)
    q_hi_norm = _norm(q_hi_phys)
    lo_final = torch.minimum(q_lo_norm, q_hi_norm)
    hi_final = torch.maximum(q_lo_norm, q_hi_norm)

    v_max_norm = None
    if v_max_phys is not None:
        scale = normalizer.params_dict["scale"][ARM_IDX].to(device, dtype)
        v_max_norm = (v_max_phys.to(device, dtype) * scale).abs()

    q_anchor_norm = None
    if q_anchor_phys is not None:
        B = q_anchor_phys.shape[0]
        dummy = torch.zeros(B, 14, device=device, dtype=dtype)
        dummy[:, ARM_IDX] = q_anchor_phys.to(device, dtype)
        q_anchor_norm = normalizer.normalize(dummy)[:, ARM_IDX]

    return lo_final, hi_final, v_max_norm, q_anchor_norm


# -----------------------------------------------------------------------
# MANDATORY next step: identity verification -- but NOT eci vs B1.
#
# eci_conditional_sample's PPR renoise step (forward-diffusion renoise
# from the projected Tweedie x0 estimate) is NOT the same formula as
# DDPMScheduler.step()'s own posterior sampling (a variance-weighted
# combination of x0_hat and x_t, not a pure forward renoise from x0_hat
# alone). This means eci_conditional_sample is NOT expected to reproduce
# policy.conditional_sample()'s output bit-for-bit even with a vacuous
# (full-range) fault_spec -- the Panda-era session notes report exactly
# this ("eci tends to score slightly lower than posthoc/B1 even under a
# vacuous fault_spec, e.g. Square -0.04"), and the ACTUAL identity gate
# used historically was posthoc_conditional_sample() vs B1 (plain
# conditional_sample()), not eci vs B1.
#
# The correct check before trusting ANY fault-sweep result from this
# module:
#   posthoc_conditional_sample(policy, ..., fault_spec=<all-healthy>)
# should be numerically identical (fixed seed) to:
#   policy.conditional_sample(...)
# because posthoc's projection is applied only once, after generation
# completes, and a vacuous (full-range) fault_spec makes that projection
# a no-op -- so posthoc reduces to exactly B1's own sampling call.
#
# eci_conditional_sample should be evaluated empirically against B1/
# posthoc on real task success rates, not via a bit-for-bit identity
# check -- some small, real divergence from B1 under vacuous conditions
# is expected.
# -----------------------------------------------------------------------
