"""
Make LIBERO datasets usable by the existing robosuite-based fault runner.

favor_fault_runner.FaultRobomimicImageRunner builds its environment from the
dataset's own env_args (FileUtils.get_env_metadata_from_dataset). For the
robosuite Lift/Can/Square datasets that works directly. For LIBERO it fails
twice, and both failures are in the metadata rather than in the runner:

1. env_name is "Libero_Floor_Manipulation" (and siblings), which robosuite
   does not know -- its registry only contains Lift, Stack, NutAssembly,
   PickPlace, Door, Wipe, ToolHang, TwoArm*. LIBERO registers its own env
   classes as a side effect of importing libero.libero.envs, so the fix is
   simply to import it before create_env runs.

2. bddl_file / bddl_file_name point at
       chiliocosm/bddl_files/<suite>/pick_the_alphabet_soup_....bddl
   "chiliocosm" is LIBERO's pre-release package name, and the recorded task
   filenames drop the word "up" that the shipped files actually carry
   ("pick_the_..." vs "pick_up_the_..."). Neither path resolves against an
   installed LIBERO. The path is therefore rebuilt from the shipped bddl
   tree by matching on the suite plus the task stem, tolerating that
   missing "up".

Both repairs are metadata-only: the controller swap to JOINT_POSITION, the
JointActuationWrapper, and the FaultInjector wrapping all happen downstream
and are untouched, so the LIBERO path exercises exactly the same fault
machinery that produced the robosuite results.
"""
import os
import re
import sys
import types


def _install_mujoco_py_stub() -> None:
    """Satisfy robomimic 0.2.0's unconditional `import mujoco_py`.

    diffusion_policy reaches the environment through
    robomimic.envs.env_robosuite.EnvRobosuite, which imports mujoco_py at
    module scope. That is fine under robosuite 1.2 (favor-p0) but fails in
    this image, where robosuite 1.4 replaced mujoco_py with the official
    `mujoco` bindings and mujoco_py is deliberately absent. LIBERO itself
    never hits this: it wraps robosuite directly and does not use
    robomimic.envs at all (verified by grep across the LIBERO source).

    mujoco_py is referenced exactly twice in all of robomimic: the import,
    and `mujoco_py.builder.MujocoException` in
    EnvRobosuite.rollout_exceptions, naming the exception type that should
    cause a rollout to be discarded. Under robosuite 1.4 that exception can
    never be raised -- physics failures surface as mujoco.FatalError -- so
    a placeholder class is behaviourally equivalent, not an approximation.

    No-op when the real mujoco_py is importable, so favor-p0 is unaffected.
    """
    if "mujoco_py" in sys.modules:
        return
    try:
        import mujoco_py  # noqa: F401  (real package present, e.g. favor-p0)
        return
    except ImportError:
        pass

    stub = types.ModuleType("mujoco_py")
    builder = types.ModuleType("mujoco_py.builder")

    class MujocoException(Exception):
        """Placeholder for the mujoco_py exception robosuite 1.4 never raises."""

    builder.MujocoException = MujocoException
    stub.builder = builder
    sys.modules["mujoco_py"] = stub
    sys.modules["mujoco_py.builder"] = builder


# Runs on import. favor_fault_runner imports this module ahead of any
# robomimic-backed import for exactly this reason.
_install_mujoco_py_stub()

def _alias_robomimic_moved_symbols() -> None:
    """Re-expose symbols robomimic 0.3.0 relocated, under their 0.2.0 names.

    This image runs robomimic 0.3.0 because 0.2.0 cannot import against
    robosuite 1.4 (it needs postprocess_model_xml, removed in 1.4). The
    upgrade moves things around, and diffusion_policy was written against
    0.2.0: diffusion_unet_hybrid_image_policy.py does
    `import robomimic.models.base_nets as rmbn` and then
    `isinstance(x, rmbn.CropRandomizer)`, but 0.3.0 moved CropRandomizer to
    robomimic.models.obs_core (base_nets keeps no Randomizer classes at all).

    Aliasing rather than editing diffusion_policy keeps the vendored
    upstream clean and keeps favor-p0 -- which still has robomimic 0.2.0 and
    the original layout -- working unchanged: the alias is only installed
    when the symbol is genuinely missing.
    """
    try:
        import robomimic.models.base_nets as base_nets
    except ImportError:
        return

    if hasattr(base_nets, "CropRandomizer"):
        return  # robomimic 0.2.0 layout (favor-p0): nothing to do

    try:
        import robomimic.models.obs_core as obs_core
    except ImportError:
        return

    for name in ("CropRandomizer", "Randomizer", "ColorRandomizer",
                 "GaussianNoiseRandomizer"):
        if hasattr(obs_core, name) and not hasattr(base_nets, name):
            setattr(base_nets, name, getattr(obs_core, name))


_alias_robomimic_moved_symbols()


LIBERO_ENV_PREFIX = "Libero"


def _libero_bddl_root() -> str:
    """Locate the shipped bddl_files directory of the installed LIBERO."""
    try:
        import libero.libero as libero_pkg
    except ImportError as exc:
        raise ImportError(
            "LIBERO dataset requires the `libero` package to be importable "
            "(it registers the Libero_* robosuite environments and ships the "
            "bddl files). Install it, or add the LIBERO checkout to PYTHONPATH."
        ) from exc

    root = os.path.join(os.path.dirname(libero_pkg.__file__), "bddl_files")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"LIBERO bddl_files not found at {root}")
    return root


def _repair_bddl_path(recorded_path: str) -> str:
    """Map a recorded (stale) bddl path onto the installed one."""
    root = _libero_bddl_root()
    suite = os.path.basename(os.path.dirname(recorded_path))
    stem = os.path.basename(recorded_path)

    suite_dir = os.path.join(root, suite)
    if not os.path.isdir(suite_dir):
        raise FileNotFoundError(
            f"bddl suite directory missing: {suite_dir} (from recorded path {recorded_path})"
        )

    exact = os.path.join(suite_dir, stem)
    if os.path.isfile(exact):
        return exact

    with_up = re.sub(r"^pick_the_", "pick_up_the_", stem)
    candidate = os.path.join(suite_dir, with_up)
    if os.path.isfile(candidate):
        return candidate

    tail = stem.replace("pick_the_", "").replace("pick_up_the_", "")
    matches = [f for f in os.listdir(suite_dir) if f.endswith(tail)]
    if len(matches) == 1:
        return os.path.join(suite_dir, matches[0])

    raise FileNotFoundError(
        f"could not resolve bddl file for {recorded_path!r} under {suite_dir} "
        f"(candidates: {matches})"
    )


def is_libero_env_meta(env_meta: dict) -> bool:
    return str(env_meta.get("env_name", "")).startswith(LIBERO_ENV_PREFIX)


def prepare_libero_env_meta(env_meta: dict) -> dict:
    """Register LIBERO envs and repair bddl paths. No-op for other datasets."""
    if not is_libero_env_meta(env_meta):
        return env_meta

    import libero.libero.envs  # noqa: F401  (registers Libero_* with robosuite)

    kwargs = env_meta.setdefault("env_kwargs", {})
    for key in ("bddl_file", "bddl_file_name"):
        recorded = kwargs.get(key) or env_meta.get(key)
        if recorded:
            fixed = _repair_bddl_path(recorded)
            if key in kwargs:
                kwargs[key] = fixed
            if key in env_meta:
                env_meta[key] = fixed

    # Force the camera resolution to match shape_meta (84x84, the same
    # resolution the robosuite lift/can/square configs render at). LIBERO's
    # bddl_base_domain.py defaults camera_heights/camera_widths to 256, and
    # env_wrapper.py overrides that to 128 -- neither matches shape_meta,
    # and RobomimicImageWrapper has no resize step of its own (it only uses
    # shape_meta to declare the observation_space; it never reshapes the
    # actual image), so images arrived at native LIBERO resolution while the
    # gym Box space said (3,84,84). That produced a downstream shape
    # mismatch inside gym's AsyncVectorEnv (concatenate() / np.stack()
    # raising "Output array is the wrong shape") that had nothing to do
    # with the concatenate() argument-order issue this file's other patches
    # address. Rendering at 84x84 directly avoids adding a resize step
    # anywhere else and matches how the robosuite tasks are already set up.
    kwargs["camera_heights"] = 84
    kwargs["camera_widths"] = 84

    print(f"[libero_env_adapter] env_name={env_meta['env_name']} "
          f"bddl={kwargs.get('bddl_file_name')} "
          f"camera={kwargs['camera_heights']}x{kwargs['camera_widths']}")
    return env_meta

