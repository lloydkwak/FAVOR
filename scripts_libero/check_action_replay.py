"""
Direct re-verification, on the CURRENT (post bddl-fix, post gripper-fix)
pipeline: does replaying the demo's OWN recorded actions through the
LIVE environment (not state-injection, not open-loop prediction) actually
reach success? This is the most basic possible sanity check and has NOT
been confirmed on bowl_stove/drawer's current bddl mapping (fixed 2026-09-14)
-- prior "predicate fires correctly" claims may predate that fix or may
have been established only for alphabet_soup and assumed to generalize.

If this succeeds: physics, controller gains, gripper action indexing, and
the success predicate are all confirmed fine under the CURRENT setup for
this task -- narrows the problem specifically to the policy/model.
If this FAILS: the problem is upstream of the policy entirely (physics/
controller/predicate under the current bddl mapping), which would be a
genuinely new finding, not yet identified.
"""
import sys, os
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import h5py
import numpy as np

from favor_fault_runner import FaultRobomimicImageRunner

TASKS = {
    "bowl_stove": {
        "dataset": "/workspace/data/robomimic/datasets/libero_bowl_stove/ph/image_abs.hdf5",
        "shape_meta": None,
    },
    "drawer": {
        "dataset": "/workspace/data/robomimic/datasets/libero_drawer/ph/image_abs.hdf5",
        "shape_meta": None,
    },
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


def replay_task(task_name, info):
    print(f"\n{'='*60}\n{task_name} -- LIVE ACTION REPLAY\n{'='*60}")

    with h5py.File(info["dataset"], "r") as f:
        demo_actions = f["data/demo_0/actions"][:]
        demo_len = len(demo_actions)
    print(f"demo_0 length: {demo_len} steps")

    runner = FaultRobomimicImageRunner(
        output_dir=f"/workspace/results/_action_replay_{task_name}",
        dataset_path=info["dataset"], shape_meta=SHAPE_META,
        fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
        n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
        n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
        abs_action=True, actuation_mode="joint", joint_kp=150,
    )
    env = runner.env.env_fns[0]()

    holder = env
    while not hasattr(holder, "init_state"):
        holder = holder.env
    with h5py.File(info["dataset"], "r") as f:
        holder.init_state = f["data/demo_0/states"][0]

    env.reset()
    best_reward = 0.0
    for t in range(demo_len):
        action = demo_actions[t][None]  # (1, 8), already joint+gripper
        _, r, done, _ = env.step(action)
        best_reward = max(best_reward, float(np.max(r)))
        if done:
            break

    verdict = "SUCCESS" if best_reward > 0.5 else "FAIL"
    print(f"live action replay of demo_0's OWN recorded actions: "
          f"max_reward={best_reward:.2f}  {verdict}")
    env.close()


import sys as _sys
target = _sys.argv[1] if len(_sys.argv) > 1 else None
for name, info in TASKS.items():
    if target is not None and name != target:
        continue
    replay_task(name, info)
