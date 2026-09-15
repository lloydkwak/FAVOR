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
}

N_TEST = 20  # Phase 1 scale (narrower than Phase 2's 50) to survey all 21 conditions cheaply first
TEST_START_SEED = 10000  # same convention as sweep_grid.py -- fixed across ALL conditions
