"""
D1 v2: same FK check, but adding the robot base's world offset before
comparing -- pytorch_kinematics computes FK with the URDF root pinned at
the origin, while the live sim places the robot base at robot0_base's
world position (confirmed: (-0.6, 0, 0), and the raw FK position error was
~0.6935 -- almost exactly matching |base_pos| once base_pos's x,y,z are
combined with the arm's own reach, not a coincidence).
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

chain = pk.build_serial_chain_from_urdf(open(URDF, "rb").read(), end_link_name="panda_hand")

runner = FaultRobomimicImageRunner(
    output_dir="/tmp/verify_fk2", dataset_path=DATASET, shape_meta=SHAPE_META,
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

addrs = [raw.sim.model.get_joint_qpos_addr("robot0_joint%d" % i) for i in range(1, 8)]
base_pos = np.array(raw.sim.data.get_body_xpos("robot0_base"))
base_rot = np.array(raw.sim.data.get_body_xmat("robot0_base")).reshape(3, 3)
print("robot0_base world pos:", np.round(base_pos, 4))

errs_pos, errs_rot = [], []
for trial in range(20):
    for _ in range(5):
        env.step(np.zeros((1, 8), dtype=np.float32))
    q = np.array([raw.sim.data.qpos[a] for a in addrs])
    q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)

    ret = chain.forward_kinematics(q_t)
    m = ret.get_matrix()
    fk_pos_local = m[0, :3, 3].numpy()
    fk_rot = m[0, :3, :3].numpy()

    # transform from robot-base-local frame into world frame
    fk_pos_world = base_rot @ fk_pos_local + base_pos
    fk_rot_world = base_rot @ fk_rot

    sim_pos = np.array(raw.sim.data.get_body_xpos("robot0_right_hand"))
    sim_rot = np.array(raw.sim.data.get_body_xmat("robot0_right_hand")).reshape(3, 3)

    pos_err = np.linalg.norm(fk_pos_world - sim_pos)
    rot_err = np.linalg.norm(fk_rot_world - sim_rot)
    errs_pos.append(pos_err)
    errs_rot.append(rot_err)

errs_pos = np.array(errs_pos)
errs_rot = np.array(errs_rot)
print()
print("position error (m):  mean=%.6f  max=%.6f" % (errs_pos.mean(), errs_pos.max()))
print("rotation error (Frobenius): mean=%.6f  max=%.6f" % (errs_rot.mean(), errs_rot.max()))
print()
print("VERDICT:", "PASS" if errs_pos.max() < 0.01 else "FAIL")
