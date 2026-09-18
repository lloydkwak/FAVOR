"""Sweep grid for the range_reduced fault sweep, mirroring sweep_grid_libero.py.

severity=0.05 chosen to match the robosuite Phase 3 convention (0.05 was
the milder of the two severities used there). Theoretically range_reduced
should be easier than locked for two reasons (see project notes): (1) it
retains everything locked has -- the same null-space reallocation via the
other 6 joints -- and (2) additionally, any candidate whose predicted
joint-j value already falls inside [q_lo, q_hi] isn't clamped at all, so
it executes with ZERO distortion, a category of "free win" locked can
never produce (locked always clamps unless the prediction hits q_lock
exactly). severity=0.05 keeps the window fairly narrow so the comparison
against locked's already-collected data stays meaningful.
"""

JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]

FAULT_CONDITIONS = [
    ("range_reduced", 0.05),
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

N_TEST = 20  # same as locked Phase 1, for a like-for-like comparison
TEST_START_SEED = 10000  # same seed convention -- same 20 episodes as locked's B1/select/random_n runs
