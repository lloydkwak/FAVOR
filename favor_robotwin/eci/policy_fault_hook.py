"""
Policy-server-side fault hook (2단계-6, B1 vs posthoc vs eci comparison).

Runs inside the POLICY SERVER process (setup_policy_server.py -> model.py's
Model class) and routes the loaded DP policy's sampling through
posthoc_conditional_sample or eci_conditional_sample.

=== WHY THIS FILE WAS REWRITTEN (v2) ===

The first version built fault_spec ONCE at server startup and, for
fault_type="locked", encoded the lock as "pin this joint to 0.0 (the
aloha-agilex homestate)". That was wrong, and it invalidated the first
Phase 2 run:

  The env-side FaultInjector locks a joint at q_onset -- the angle the
  joint actually happened to be at when the fault engaged -- NOT at 0.0.
  q_onset varies by task, seed, and when in the episode the lock takes
  hold. So the policy was being told "drive this joint to 0.0" while the
  real robot had that joint frozen somewhere else entirely: the policy's
  constraint and the physical constraint described different worlds.

  Worse, this bad constraint hit posthoc and eci ASYMMETRICALLY. posthoc
  applies the projection once, at the very end; eci applies it at every
  one of the ~100 denoising steps. So a wrong q_lo/q_hi contaminates eci
  ~100x more than posthoc. The resulting "posthoc and eci disagree about
  which conditions they help" pattern in Phase 2 run #1 is therefore not
  evidence about PPR-renoise vs single-clip at all -- it is mostly an
  artifact of one method being exposed to the same modeling error far
  more often than the other.

=== HOW v2 FIXES IT ===

fault_spec is now rebuilt PER INFERENCE CALL from the robot's actual
observed joint configuration, by wrapping policy.predict_action (not
policy.conditional_sample). predict_action receives the raw obs_dict,
which contains 'agent_pos' -- the normalized 14-dim joint state the
policy already conditions on, i.e. exactly the live robot state this
process previously lacked. For a locked fault we read that joint's
current observed value and use it as q_lo == q_hi, matching
FaultInjector's q_onset semantics without needing any new cross-process
channel.

Note agent_pos arrives ALREADY NORMALIZED in the same LinearNormalizer
space the projector operates in, so for locked faults no physical<->
normalized conversion is needed at all -- we read the normalized value and
constrain in normalized space directly. This also sidesteps the previous
version's dependence on hand-copied physical joint limits for the locked
case. range_reduced still needs the physical table (it is defined as a
fraction of physical range), and for that case the home-centered window
DOES match FaultInjector, which centers on config.yml's homestate (0.0
for every aloha-agilex arm joint) rather than on q_onset -- so that path
is unchanged from v1 and was never affected by this bug.

Configuration (from model_cfg, i.e. deploy.yml + --overrides):
    fault_arm       "left" | "right"
    fault_joint     e.g. "fl_joint4"
    fault_type      "locked" | "range_reduced" | "velocity_limited"
    fault_severity  float in (0,1], for range_reduced/velocity_limited
    sampling_mode   "b1" (default) | "posthoc" | "eci"
    eci_n_resample  int, only for sampling_mode="eci" (default 1)

sampling_mode="b1" leaves the policy completely untouched.
"""
import torch

from diffusion_policy.eci.joint_eci_projector_robotwin import (
    posthoc_conditional_sample, eci_conditional_sample, normalize_joint_bounds, ARM_IDX,
)

LEFT_ARM_JOINTS = ["fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6"]
RIGHT_ARM_JOINTS = ["fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6"]

# Physical ranges, hand-synced from envs/fault_injection/fault_injector_sapien.py's
# JOINT_LIMITS_RAD (not importable across the process boundary). Only used for
# range_reduced / velocity_limited; the locked path works purely in normalized
# space off the observation and needs none of this.
PER_ARM_LIMITS_RAD = {
    "joint1": None,
    "joint2": (0.0, 2.5121),
    "joint3": (0.0, 2.4937),
    "joint4": (-1.6074, 1.5284),
    "joint5": (-0.8292, 0.8490),
    "joint6": (-5.5697, 3.4248),
}

PER_ARM_VEL_MAX_RAD_S = {
    "joint1": 0.598, "joint2": 1.611, "joint3": 1.386,
    "joint4": 1.572, "joint5": 0.618, "joint6": 1.771,
}


def _generic_joint_key(joint_name: str) -> str:
    for prefix in ("fl_", "fr_"):
        if joint_name.startswith(prefix):
            return joint_name[len(prefix):]
    raise ValueError(f"Unrecognized joint name '{joint_name}' (expected fl_/fr_ prefix)")


def _arm_idx_position(arm_tag: str, joint_name: str) -> int:
    """Position of this joint within the 12-wide [left(6), right(6)] ARM_IDX ordering."""
    joint_list = LEFT_ARM_JOINTS if arm_tag == "left" else RIGHT_ARM_JOINTS
    assert joint_name in joint_list, f"{joint_name} not valid for arm_tag={arm_tag}"
    local_idx = joint_list.index(joint_name)
    return local_idx if arm_tag == "left" else 6 + local_idx


def _action_dim_index(arm_tag: str, joint_name: str) -> int:
    """Index of this joint within the raw 14-dim action/state vector
    ([left_arm(6), left_gripper, right_arm(6), right_gripper])."""
    joint_list = LEFT_ARM_JOINTS if arm_tag == "left" else RIGHT_ARM_JOINTS
    local_idx = joint_list.index(joint_name)
    return local_idx if arm_tag == "left" else 7 + local_idx


def _build_static_fault_spec(policy, arm_tag, joint_name, fault_type, severity):
    """For range_reduced / velocity_limited: a fault_spec that does NOT
    depend on live robot state (range_reduced is a fixed home-centered
    window, matching FaultInjector). Returns None for locked, which is
    built per-call instead."""
    if fault_type == "locked":
        return None

    generic_key = _generic_joint_key(joint_name)
    global_idx = _arm_idx_position(arm_tag, joint_name)
    all_joints = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

    lo_phys = torch.tensor([
        (PER_ARM_LIMITS_RAD.get(_generic_joint_key(j)) or (-10.0, 10.0))[0] for j in all_joints
    ])
    hi_phys = torch.tensor([
        (PER_ARM_LIMITS_RAD.get(_generic_joint_key(j)) or (-10.0, 10.0))[1] for j in all_joints
    ])

    v_max_phys = None
    q_anchor_phys = None

    if fault_type == "range_reduced":
        assert severity is not None and 0.0 < severity <= 1.0
        lims = PER_ARM_LIMITS_RAD.get(generic_key)
        if lims is None:
            raise ValueError(f"range_reduced on {joint_name}: no trustworthy physical range "
                              f"for {generic_key} (joint1 excluded)")
        lo, hi = lims
        home = 0.0  # matches FaultInjector's homestate-centered window
        half = 0.5 * (hi - lo) * severity
        lo_phys[global_idx] = max(lo, home - half)
        hi_phys[global_idx] = min(hi, home + half)

    elif fault_type == "velocity_limited":
        assert severity is not None and 0.0 < severity <= 1.0
        vmax = PER_ARM_VEL_MAX_RAD_S.get(generic_key)
        v_max_phys = torch.full((12,), 1e6)
        v_max_phys[global_idx] = vmax * severity
        # q_anchor is filled in per-call from the live observation (see
        # the wrapper below) -- a zeros placeholder here would reintroduce
        # exactly the v1 bug for the velocity path.
    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    lo_norm, hi_norm, v_max_norm, _ = normalize_joint_bounds(
        policy, lo_phys, hi_phys, v_max_phys=v_max_phys, q_anchor_phys=None,
    )
    spec = {"q_lo": lo_norm, "q_hi": hi_norm}
    if v_max_norm is not None:
        spec["v_max"] = v_max_norm
    return spec


def apply_policy_fault_hook(policy, model_cfg: dict) -> None:
    """Wrap policy.predict_action so each inference call builds a
    fault_spec reflecting the CURRENT observed robot state, then routes
    sampling through posthoc/eci. No-op under sampling_mode='b1'."""
    sampling_mode = str(model_cfg.get("sampling_mode", "b1")).lower()
    if sampling_mode == "b1":
        return

    arm = model_cfg.get("fault_arm")
    joint = model_cfg.get("fault_joint")
    ftype = model_cfg.get("fault_type")
    if not arm or not joint or not ftype:
        print(f"[POLICY_FAULT_HOOK] sampling_mode={sampling_mode} but no fault params "
              f"-- leaving policy unpatched (running as B1).", flush=True)
        return

    severity = model_cfg.get("fault_severity")
    severity = float(severity) if severity is not None else None
    n_resample = int(model_cfg.get("eci_n_resample", 1))

    static_spec = _build_static_fault_spec(policy, arm, joint, ftype, severity)
    arm_pos = _arm_idx_position(arm, joint)        # 0..11, within ARM_IDX order
    act_pos = _action_dim_index(arm, joint)        # 0..13, within raw action vector

    original_predict_action = policy.predict_action
    original_conditional_sample = policy.conditional_sample

    def patched_predict_action(obs_dict):
        # agent_pos: (B, To, 14) normalized joint state -- the live robot
        # configuration this process previously had no access to.
        agent_pos = obs_dict["agent_pos"]
        latest = agent_pos[:, -1, :]  # (B, 14), most recent observed state

        if ftype == "locked":
            # BUG FOUND AND FIXED (v3): obs_dict['agent_pos'] arrives at
            # predict_action() in PHYSICAL units (radians) -- predict_action
            # itself does `nobs = self.normalizer.normalize(obs_dict)` as
            # its own first internal step, meaning what THIS wrapper
            # receives is pre-normalization. v2 used that raw physical
            # value directly as q_lo/q_hi against a projector that operates
            # on the NORMALIZED trajectory (x0_hat, already clamped to
            # [-1,1] via scheduler.config.clip_sample) -- forcing that
            # joint dimension to a value routinely 1-3+ units outside the
            # valid normalized range on every application. Confirmed via a
            # real run's debug log showing agent_pos entries like 2.512,
            # 2.025 (physically real, impossible in normalized space).
            # This actively corrupted the trajectory rather than
            # constraining it, worse for eci (~100 applications per call)
            # than posthoc (1 application) -- matching the observed
            # "eci consistently worse than posthoc, both worse than B1"
            # pattern in the v2 Phase 2 run.
            #
            # Fix: run the observed physical value through the SAME
            # normalize_joint_bounds() helper already used (and validated)
            # for the range_reduced/velocity_limited paths, so the bound
            # actually lives in the space the projector operates in.
            B = latest.shape[0]
            locked_val_phys = latest[:, act_pos]  # (B,), physical radians

            # Build (B,12) physical lo=hi vectors: the locked joint gets
            # its real observed value, every other position is an
            # arbitrary placeholder (never used -- overwritten below by
            # the wide-open identity bound).
            lo_phys_batch = torch.zeros(B, 12, device=latest.device, dtype=latest.dtype)
            hi_phys_batch = torch.zeros(B, 12, device=latest.device, dtype=latest.dtype)
            lo_phys_batch[:, arm_pos] = locked_val_phys
            hi_phys_batch[:, arm_pos] = locked_val_phys

            lo_norm_batch, hi_norm_batch, _, _ = normalize_joint_bounds(
                policy, lo_phys_batch, hi_phys_batch,
            )

            q_lo = torch.full((B, 12), -1e6, device=latest.device, dtype=latest.dtype)
            q_hi = torch.full((B, 12), 1e6, device=latest.device, dtype=latest.dtype)
            q_lo[:, arm_pos] = lo_norm_batch[:, arm_pos]
            q_hi[:, arm_pos] = hi_norm_batch[:, arm_pos]
            fault_spec = {"q_lo": q_lo, "q_hi": q_hi}

        else:
            fault_spec = dict(static_spec)
            if "v_max" in fault_spec:
                # velocity recursion must start from where the arm really
                # is, not from zeros
                fault_spec["q_anchor"] = latest[:, ARM_IDX.to(latest.device)]

        def sampler(condition_data, condition_mask, **kwargs):
            # posthoc/eci call policy.conditional_sample internally and
            # expect the ORIGINAL unconstrained sampler there. Restore it
            # for the duration of that inner call, then put `sampler` back
            # so any further calls in this same predict_action still route
            # through the projector. Without this the inner call re-enters
            # `sampler` -> infinite recursion (hit for real in v1).
            policy.conditional_sample = original_conditional_sample
            try:
                if sampling_mode == "posthoc":
                    return posthoc_conditional_sample(
                        policy, condition_data, condition_mask, fault_spec, **kwargs)
                else:
                    return eci_conditional_sample(
                        policy, condition_data, condition_mask, fault_spec,
                        n_resample=n_resample, **kwargs)
            finally:
                policy.conditional_sample = sampler

        # posthoc/eci call policy.conditional_sample internally expecting
        # the ORIGINAL sampler; swap in the original for the duration of
        # the call, then restore, so there is no recursion (this bug was
        # already hit once in v1 and surfaced as "maximum recursion depth
        # exceeded" through the websocket).
        policy.conditional_sample = sampler
        try:
            return original_predict_action(obs_dict)
        finally:
            policy.conditional_sample = original_conditional_sample

    policy.predict_action = patched_predict_action
    print(f"[POLICY_FAULT_HOOK] v2 active: sampling_mode={sampling_mode} "
          f"arm={arm} joint={joint} type={ftype} severity={severity} "
          f"(fault_spec rebuilt per-call from observed agent_pos)", flush=True)
