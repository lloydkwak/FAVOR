"""
Identity verification for the RoboTwin/DP E-C-I port.

Confirms: under a fixed generator seed, posthoc_conditional_sample() with
a vacuous fault_spec (bounds covering the full physical range on every
arm joint, no velocity limit) produces numerically identical actions to
the original, unconstrained policy.conditional_sample(). This is the
correct identity gate (NOT eci vs B1 -- see joint_eci_projector_robotwin.py
module docstring for why eci is expected to diverge slightly even under
vacuous conditions).

Run from XPolicyLab/policy/DP:
    python diffusion_policy/eci/test_identity.py
"""
import sys
sys.path.append("./")

import torch

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.eci.joint_eci_projector_robotwin import (
    posthoc_conditional_sample, normalize_joint_bounds, ARM_IDX,
)

CKPT_PATH = "checkpoints/demo_clean-grab_roller-aloha_agilex-joint-0/600.ckpt"

PER_ARM_LIMITS_RAD = [
    # SINGLE SOURCE OF TRUTH: envs/fault_injection/fault_injector_sapien.py's
    # JOINT_LIMITS_RAD (RoboTwin repo root, not importable from here since
    # eval.sh runs the DP policy and the SAPIEN env as separate processes
    # over a websocket -- these values are copied by hand and must be kept
    # in sync manually whenever that file's JOINT_LIMITS_RAD changes).
    # Last synced: empirical values from 23,008 frames across grab_roller +
    # adjust_bottle + handover_mic demo_clean data.
    #
    # joint1 keeps a wide placeholder [-10, 10] here (not the empirical
    # [-7.34, 0.0]) because this table's only use in this file is building
    # a vacuous (fully-permissive) fault_spec for the identity check --
    # joint1 is excluded from any real range_reduced fault, so it only
    # needs a bound wide enough to guarantee no clipping occurs here.
    (-10.0, 10.0),        # joint1 -- placeholder, excluded from real range_reduced faults
    (0.0, 2.5121),          # joint2
    (0.0, 2.4937),          # joint3
    (-1.6074, 1.5284),      # joint4
    (-0.8292, 0.8490),      # joint5
    (-5.5697, 3.4248),      # joint6 (wide -- see fault_injector_sapien.py caveat)
]
FULL_RANGE_LO = torch.tensor([lo for lo, hi in PER_ARM_LIMITS_RAD] * 2)
FULL_RANGE_HI = torch.tensor([hi for lo, hi in PER_ARM_LIMITS_RAD] * 2)


def load_policy(ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    workspace = RobotWorkspace(cfg)
    workspace.load_payload(payload)
    policy = workspace.model
    policy.eval()
    return policy


def make_dummy_condition(policy, batch_size=1):
    device = policy.device
    dtype = policy.dtype
    T = policy.horizon
    Da = policy.action_dim
    B = batch_size
    cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    global_cond_dim = policy.obs_feature_dim * policy.n_obs_steps
    global_cond = torch.zeros(size=(B, global_cond_dim), device=device, dtype=dtype)
    return cond_data, cond_mask, global_cond


def main():
    print(f"Loading policy from {CKPT_PATH} ...")
    policy = load_policy(CKPT_PATH)

    cond_data, cond_mask, global_cond = make_dummy_condition(policy)

    # For THIS identity check, bypass physical-unit conversion entirely:
    # test the project_fault/posthoc plumbing using bounds that cover the
    # network's own valid output range directly in NORMALIZED space
    # ([-1, 1], enforced by scheduler.config.clip_sample on x0_hat).
    # Physical-radian -> normalized conversion is only needed for REAL
    # fault experiments (FaultInjector reports faults in physical rad);
    # it is a separate, still-open question (see joint6 discrepancy) that
    # should not block verifying the ARM_IDX/clamp plumbing itself.
    vacuous_fault_spec = {
        "q_lo": torch.full((12,), -1.0),
        "q_hi": torch.full((12,), 1.0),
    }

    seed = 12345

    print("Running original conditional_sample (B1) ...")
    gen_a = torch.Generator(device=policy.device).manual_seed(seed)
    with torch.no_grad():
        traj_a = policy.conditional_sample(
            cond_data, cond_mask, local_cond=None, global_cond=global_cond, generator=gen_a,
        )

    print("Running posthoc_conditional_sample with vacuous fault_spec ...")
    gen_b = torch.Generator(device=policy.device).manual_seed(seed)
    with torch.no_grad():
        traj_b = posthoc_conditional_sample(
            policy, cond_data, cond_mask, fault_spec=vacuous_fault_spec,
            local_cond=None, global_cond=global_cond, generator=gen_b,
        )

    diff = (traj_a - traj_b).abs()
    print(f"\nmax abs diff (full 14-dim trajectory): {diff.max().item():.8f}")
    print(f"mean abs diff: {diff.mean().item():.8f}")

    arm_diff = diff[..., ARM_IDX]
    print(f"max abs diff (arm joints only, ARM_IDX): {arm_diff.max().item():.8f}")

    gripper_idx = torch.tensor([6, 13])
    gripper_diff = diff[..., gripper_idx]
    print(f"max abs diff (gripper indices 6,13): {gripper_diff.max().item():.8f}")

    TOL = 1e-5
    if diff.max().item() < TOL:
        print(f"\nPASS -- posthoc matches B1 within tolerance {TOL}")
    else:
        print(f"\nFAIL -- outputs diverge beyond tolerance {TOL}. Root-cause before "
              f"trusting any fault-sweep result (likely candidates: ARM_IDX ordering "
              f"mismatch, normalizer scale/offset misuse).")


if __name__ == "__main__":
    main()
