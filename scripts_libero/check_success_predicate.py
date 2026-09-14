"""
Does _check_success() ever fire for bowl_stove and drawer, when the
environment is driven directly through the demo's own recorded state
sequence (bypassing the controller and the policy entirely)?

This isolates the success predicate from everything else: if it never
fires even on a real successful demonstration, the bddl goal condition
itself is broken (likely referencing an object/joint name that doesn't
match what actually got instantiated), not the trained policy.

Mirrors the alphabet_soup diagnostic that found demo replay succeeds at
step 121 with reward 1.0 -- the same check for the two tasks that trained
cleanly (loss converged normally) but scored 0% at every rollout.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import h5py
import numpy as np

from favor_fault_runner import FaultRobomimicImageRunner

TASKS = {
    "bowl_stove": "/workspace/data/robomimic/datasets/libero_bowl_stove/ph/image_abs.hdf5",
    "drawer": "/workspace/data/robomimic/datasets/libero_drawer/ph/image_abs.hdf5",
}

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


def raw_of(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def check(name, dataset):
    runner = FaultRobomimicImageRunner(
        output_dir="/tmp/check_success", dataset_path=dataset, shape_meta=SHAPE_META,
        fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
        n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
        n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
        abs_action=True, actuation_mode="joint",
    )
    env = runner.env.env_fns[0]()
    r = raw_of(env)

    with h5py.File(dataset, "r") as f:
        states = f["data/demo_0/states"][:]

    succeeded_at = None
    for t in range(len(states)):
        r.sim.set_state_from_flattened(states[t])
        r.sim.forward()
        if r._check_success():
            succeeded_at = t
            break

    if succeeded_at is not None:
        print("%s: _check_success() TRUE at demo step %d / %d -- predicate OK"
              % (name, succeeded_at, len(states)))
    else:
        print("%s: _check_success() NEVER true across all %d demo states -- predicate BROKEN"
              % (name, len(states)))
        final_reward = r.reward()
        print("  final-state reward(): %s" % final_reward)


for name, dataset in TASKS.items():
    check(name, dataset)
