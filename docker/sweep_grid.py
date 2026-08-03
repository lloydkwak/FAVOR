"""Single source of truth for the Phase 3 sweep grid."""

JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]

# (fault_type, severity) — None severity for locked (no parameter needed)
FAULT_CONDITIONS = [
    ("locked", None),
    ("range_reduced", 0.05),
    ("range_reduced", 0.03),
    ("velocity_limited", 0.5),
    ("velocity_limited", 0.25),
]

TASKS = {
    "lift": {
        "ckpt": "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt",
        "dataset": "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5",
    },
    "can": {
        "ckpt": "/workspace/data/checkpoints/can_ph/data/experiments/image/can_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt",
        "dataset": "/workspace/diffusion_policy/data/robomimic/datasets/can/ph/image_abs.hdf5",
    },
    "square": {
        "ckpt": "/workspace/data/checkpoints/square_ph/data/experiments/image/square_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt",
        "dataset": "/workspace/diffusion_policy/data/robomimic/datasets/square/ph/image_abs.hdf5",
    },
}

N_TEST = 50
TEST_START_SEED = 10000  # fixed across ALL conditions -> same 50 seeds everywhere,
                          # which is what makes paired McNemar comparison valid later.
