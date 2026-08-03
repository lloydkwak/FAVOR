"""Phase 0 DoD: official diffusion_policy repo imports cleanly in this env."""
import sys

FAILURES = 0

def check(name, fn):
    global FAILURES
    try:
        result = fn()
        print(f"[PASS] {name}: {result}")
    except Exception as e:
        FAILURES += 1
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def check_torch_cuda():
    import torch
    return f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"


def check_mujoco_py():
    import mujoco_py
    return f"mujoco_py {mujoco_py.__version__}"


def check_robosuite():
    import robosuite
    env = robosuite.make(
        "Lift", robots="Panda", has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False,
    )
    env.reset()
    env.close()
    return f"robosuite {robosuite.__version__} (Lift env create+reset OK)"


def check_robomimic():
    import robomimic
    return f"robomimic {robomimic.__version__}"


def check_dp_repo_import():
    sys.path.insert(0, "/workspace/diffusion_policy")
    import diffusion_policy  # noqa: F401
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,  # noqa: F401
    )
    from diffusion_policy.env_runner.robomimic_image_runner import (
        RobomimicImageRunner,  # noqa: F401
    )
    return "diffusion_policy repo import OK (image policy + RobomimicImageRunner)"


if __name__ == "__main__":
    print("=" * 72)
    print("FAVOR Phase 0 — environment verification")
    print("=" * 72)
    check("python / torch / CUDA", check_torch_cuda)
    check("mujoco_py import", check_mujoco_py)
    check("robosuite env create+reset", check_robosuite)
    check("robomimic import", check_robomimic)
    check("diffusion_policy repo import (host-mounted)", check_dp_repo_import)
    print("=" * 72)
    if FAILURES:
        print(f"{FAILURES} FAILURE(S) — Phase 0 DoD NOT met.")
        sys.exit(1)
    print("ALL PASS — Phase 0 DoD met. Next: Phase 1 (checkpoint download).")
