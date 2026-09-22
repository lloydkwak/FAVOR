"""Sweep grid for the LIBERO Phase 1 fault sweep (B1 baseline only).

Mirrors sweep_grid.py's structure and TEST_START_SEED convention (fixed
across all conditions so later paired comparisons, e.g. against a
selection-based method, are valid), scoped to the 3 LIBERO tasks whose
retraining succeeded after the gripper-index fix (bowl_stove and drawer
are excluded pending a separate closed-loop-rollout investigation --
see TODO in project notes).

Phase 1 starts with locked-only, matching how the RoboTwin sweep was
staged (locked first to find which joint/task combinations are even
worth extending to range_reduced/velocity_limited).
"""

JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]

FAULT_CONDITIONS = [
    ("locked", None),
]

TASKS = {
    "alphabet_soup": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_alphabet_soup/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_alphabet_soup/ph/image_abs.hdf5",
    },
    "milk": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_milk/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_milk/ph/image_abs.hdf5",
    },
    "bowl_ramekin": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_bowl_ramekin/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_bowl_ramekin/ph/image_abs.hdf5",
    },
    # Added after fixing the LIBERO fixture-site placement bug in
    # favor_fault_runner.py (soft reset applied the fixture transform twice, so
    # objects on the stove/cabinet spawned ~1 m above the table -> 0% before the
    # fix). Checkpoint = epoch 150 (latest), no-fault 0.85 at kp=150, n=20.
    "bowl_stove": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_bowl_stove/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_bowl_stove/ph/image_abs.hdf5",
        "joint_kp": 150,
    },
    # Articulated, contact-rich task: pulling the drawer against its resistance
    # needs higher joint stiffness than free-space pick-and-place. Demo-action
    # replay: kp=150 0/5, kp>=300 3/5; policy (epoch-50 ckpt, n=20): kp150 0.00,
    # kp300 0.15, kp600 0.50. PROVISIONAL kp -- finalize after the resumed-training
    # checkpoint is evaluated. Do not start drawer sweeps before that.
    "drawer": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_drawer/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_drawer/ph/image_abs.hdf5",
        "joint_kp": 600,
    },
}

N_TEST = 20  # Phase 1 scale (narrower than Phase 2's 50) to survey all 21 conditions cheaply first
TEST_START_SEED = 10000  # same convention as sweep_grid.py -- fixed across ALL conditions
