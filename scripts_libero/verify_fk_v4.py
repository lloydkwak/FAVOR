"""
D1 v4: v3 found the x-offset matches perfectly but z is off by exactly
0.3335 and y by 0.1 -- not matching any single body's xpos tried so far.
Rather than guess further, dump the full body tree with world positions so
the actual parent chain (and any body between the world and robot0_base
carrying a mount-height offset) is visible directly.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import numpy as np

from favor_fault_runner import FaultRobomimicImageRunner

DATASET = "/workspace/data/robomimic/datasets/libero_alphabet_soup/ph/image_abs.hdf5"
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
    output_dir="/tmp/verify_fk4", dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint",
)
env = runner.env.env_fns[0]()
env.reset()
raw = env
while hasattr(raw, "env"):
    raw = raw.env

m = raw.sim.model
d = raw.sim.data

print("=== full body tree: id, name, parent_id, parent_name, world xpos ===")
for i in range(m.nbody):
    name = m.body_id2name(i)
    parent_id = m.body_parentid[i]
    parent_name = m.body_id2name(parent_id) if parent_id >= 0 else "(world)"
    pos = d.xpos[i]
    if name and ("robot0" in name):
        print("%3d  %-25s  parent=%-3d %-25s  xpos=%s" % (i, name, parent_id, parent_name, np.round(pos, 4)))
