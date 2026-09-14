"""
D1 v3: debug the remaining offset by checking FK against sim with q close
to the URDF's own zero configuration, and printing every intermediate
value instead of just the final error -- v2's naive base_rot @ fk_pos +
base_pos halved the error (0.693 -> 0.348) rather than zeroing it, which
means either base_rot is being applied in the wrong direction/order, or an
additional fixed offset (e.g. a mount height in the MJCF not present in
the URDF) is still unaccounted for.
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
    output_dir="/tmp/verify_fk3", dataset_path=DATASET, shape_meta=SHAPE_META,
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
link0_pos = np.array(raw.sim.data.get_body_xpos("robot0_link0"))
link0_rot = np.array(raw.sim.data.get_body_xmat("robot0_link0")).reshape(3, 3)

print("robot0_base  world pos:", np.round(base_pos, 4), "rot diag:", np.round(np.diag(base_rot), 4))
print("robot0_link0 world pos:", np.round(link0_pos, 4), "rot diag:", np.round(np.diag(link0_rot), 4))

q = np.array([raw.sim.data.qpos[a] for a in addrs])
q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
ret = chain.forward_kinematics(q_t)
m = ret.get_matrix()
fk_pos_local = m[0, :3, 3].numpy()
fk_rot_local = m[0, :3, :3].numpy()
print("FK local pos (URDF root frame):", np.round(fk_pos_local, 4))

sim_pos = np.array(raw.sim.data.get_body_xpos("robot0_right_hand"))
sim_rot = np.array(raw.sim.data.get_body_xmat("robot0_right_hand")).reshape(3, 3)
print("sim world pos (robot0_right_hand):", np.round(sim_pos, 4))

print()
print("--- candidate corrections ---")
print("A) link0_pos + fk_local:            ", np.round(link0_pos + fk_pos_local, 4),
      " err=", np.round(np.linalg.norm(link0_pos + fk_pos_local - sim_pos), 4))
print("B) link0_pos + link0_rot @ fk_local: ", np.round(link0_pos + link0_rot @ fk_pos_local, 4),
      " err=", np.round(np.linalg.norm(link0_pos + link0_rot @ fk_pos_local - sim_pos), 4))
print("C) base_pos + fk_local:              ", np.round(base_pos + fk_pos_local, 4),
      " err=", np.round(np.linalg.norm(base_pos + fk_pos_local - sim_pos), 4))
