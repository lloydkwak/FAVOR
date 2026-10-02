"""
Locked-fault grid for the LIBERO sweeps: 4 tasks x 7 joints, n_test=20, fixed seeds
(TEST_START_SEED=10000 for every condition and method, so episodes can be paired).
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
}

N_TEST = 20
TEST_START_SEED = 10000  # fixed across ALL conditions and methods
