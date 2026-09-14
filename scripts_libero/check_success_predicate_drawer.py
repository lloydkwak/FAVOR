"""
Same check as check_success_predicate.py, but only for drawer, run as its
own process. Running both tasks in one process deadlocked on the second
environment construction (0% CPU, no progress) -- the same pattern seen
every time a second FaultRobomimicImageRunner / env_fns[0]() call happens
in the same process. One process per task, invoked from the shell, is the
only pattern that has reliably worked.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import h5py

from favor_fault_runner import FaultRobomimicImageRunner

DATASET = "/workspace/data/robomimic/datasets/libero_drawer/ph/image_abs.hdf5"
SHAPE_META = {
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3]},
        "robot0_eef_quat": {"shape": [4]},
        "robot0_gripper_qpos": {"shape": [2]},
    },
    "action": {"shape": [8]},
}

runner = FaultRobomimicImageRunner(
    output_dir="/tmp/check_success_drawer", dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint",
)
env = runner.env.env_fns[0]()
r = env
while hasattr(r, "env"):
    r = r.env

with h5py.File(DATASET, "r") as f:
    states = f["data/demo_0/states"][:]

succeeded_at = None
for t in range(len(states)):
    r.sim.set_state_from_flattened(states[t])
    r.sim.forward()
    if r._check_success():
        succeeded_at = t
        break

if succeeded_at is not None:
    print("drawer: _check_success() TRUE at demo step %d / %d -- predicate OK" % (succeeded_at, len(states)))
else:
    print("drawer: _check_success() NEVER true across all %d demo states -- predicate BROKEN" % len(states))
    print("  final-state reward(): %s" % r.reward())
