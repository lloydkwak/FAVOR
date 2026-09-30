"""
Sweep grid for the range_reduced fault sweep: 4 tasks x 7 joints x 3 levels,
n_test=20, same seeds as the locked sweep (TEST_START_SEED=10000).

fault_injector.py: window = q_onset +/- 0.5 * nominal_range * severity.
One nominal-fraction severity is not comparable across joints here: in the
demos joints 1/3/5 move only ~2-6% of their nominal range away from q_onset,
joints 2/4 move 20-55% (scripts_libero/range_severity_design.py). The old
severity=0.05 kept ~74% of milk/j1's needed motion but ~5% of bowl_stove/j4's.

So severity is set per (task, joint): each level keeps a fixed fraction
keep_frac of that joint's median demonstrated excursion max_t|q(t)-q(0)|
(median over the task's 50 demos) reachable:
    half_width = keep_frac * median_excursion
    severity   = 2 * half_width / nominal_range     (fault_injector units)
Levels (listed in run order): moderate 50%, severe 25%, mild 75%.
Together with locked (0%) and no fault (100%) this is one severity axis.
"""
import h5py
import numpy as np

JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
FAULT_TYPE = "range_reduced"
LEVELS = [("moderate", 0.50), ("severe", 0.25), ("mild", 0.75)]

# Panda nominal joint range (hi - lo); same limits as the robosuite Panda model
PANDA_RANGE = np.array([5.7946, 3.5256, 5.7946, 3.0020, 5.7946, 3.7700, 5.7946])

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
    "bowl_stove": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_bowl_stove/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_bowl_stove/ph/image_abs.hdf5",
    },
}

N_TEST = 20
TEST_START_SEED = 10000

_EXC = {}
def demo_excursion(task_name):
    """median over demos of max_t |q(t) - q(0)|, per joint [rad]"""
    if task_name not in _EXC:
        with h5py.File(TASKS[task_name]["dataset"], "r") as f:
            exc = []
            for k in f["data"].keys():
                q = f[f"data/{k}/obs/robot0_joint_pos"][:]
                exc.append(np.abs(q - q[0]).max(axis=0))
        _EXC[task_name] = np.median(np.array(exc), axis=0)
    return _EXC[task_name]

def severity_for(task_name, joint_name, keep_frac):
    j = int(joint_name.replace("robot0_joint", "")) - 1
    return float(2.0 * keep_frac * demo_excursion(task_name)[j] / PANDA_RANGE[j])
