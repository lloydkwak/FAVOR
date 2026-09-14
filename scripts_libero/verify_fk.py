"""
D1: Does pytorch_kinematics FK on robosuite's own Panda URDF match the
live simulator's reported robot0_eef_pos / robot0_eef_quat?

This is a required precondition for the null-space-steering design: every
downstream piece (the feasibility certificate, the null-space compensator)
computes Jacobians and forward kinematics OUTSIDE the simulator, so if that
computation doesn't agree with what the simulator itself considers the EE
pose to be, nothing built on top of it can be trusted.

Uses robosuite's own panda_arm_hand.urdf (not RoboTwin's or curobo's copy)
since that's the one robosuite's own IK/collision code is built against,
and its joint names (panda_joint1..7) already match ours in count and order
(robot0_joint1..7), just with a different prefix.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import numpy as np
import torch
import pytorch_kinematics as pk

from favor_fault_runner import FaultRobomimicImageRunner

URDF = "/opt/conda/envs/libero_dp/lib/python3.8/site-packages/robosuite/models/assets/bullet_data/panda_description/urdf/panda_arm_hand.urdf"
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

# Build the kinematic chain up to the hand frame (EE), not the fingers --
# robot0_eef_pos in robosuite is the hand frame, not a fingertip.
chain = pk.build_serial_chain_from_urdf(open(URDF, "rb").read(), end_link_name="panda_hand")
print("chain joint names (order pytorch_kinematics expects):", chain.get_joint_parameter_names())

runner = FaultRobomimicImageRunner(
    output_dir="/tmp/verify_fk", dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint",
)
env = runner.env.env_fns[0]()
obs = env.reset()
raw = env
while hasattr(raw, "env"):
    raw = raw.env

addrs = [raw.sim.model.get_joint_qpos_addr("robot0_joint%d" % i) for i in range(1, 8)]

errs_pos, errs_rot = [], []
for trial in range(20):
    # step a few times to get varied configurations, not just the reset pose
    for _ in range(5):
        env.step(np.zeros((1, 8), dtype=np.float32))
    q = np.array([raw.sim.data.qpos[a] for a in addrs])
    q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)

    ret = chain.forward_kinematics(q_t)
    m = ret.get_matrix()  # (1, 4, 4)
    fk_pos = m[0, :3, 3].numpy()
    fk_rot = m[0, :3, :3].numpy()

    sim_pos = np.array(raw.sim.data.get_body_xpos("robot0_right_hand"))
    sim_rot = np.array(raw.sim.data.get_body_xmat("robot0_right_hand")).reshape(3, 3)

    pos_err = np.linalg.norm(fk_pos - sim_pos)
    rot_err = np.linalg.norm(fk_rot - sim_rot)  # crude Frobenius diff
    errs_pos.append(pos_err)
    errs_rot.append(rot_err)

errs_pos = np.array(errs_pos)
errs_rot = np.array(errs_rot)
print()
print("position error (m):  mean=%.6f  max=%.6f" % (errs_pos.mean(), errs_pos.max()))
print("rotation error (Frobenius): mean=%.6f  max=%.6f" % (errs_rot.mean(), errs_rot.max()))
print()
print("VERDICT:", "PASS" if errs_pos.max() < 0.01 else "FAIL",
      "(threshold: 1cm position; a fixed offset here likely means a base-frame or link-naming mismatch, not random error)")

# ---- follow-up: is the error explained by a fixed robot-base offset? ----
print()
print("=== checking if error is a constant robot-base offset ===")
base_pos = np.array(raw.sim.data.get_body_xpos("robot0_base"))
print("robot0_base world position:", np.round(base_pos, 4))
