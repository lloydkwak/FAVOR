"""
D1 final: FK verification using curobo's franka_panda.urdf, confirmed
correct (panda_link1 at q=0 gives z=0.333, matching the live sim's
robot0_link1 body position exactly -- robosuite's own bullet_data
panda_arm_hand.urdf gave z=0 there, so it was unusable for real FK
despite being the "official" robosuite copy).

Two-stage check, to isolate errors instead of guessing at a combined
transform:

STAGE 1: local-frame comparison. Compare FK(q) computed in the URDF's own
root-link frame against the sim's robot0_right_hand position expressed
RELATIVE TO robot0_base (i.e. subtract out the base's world pose first, so
this only tests the URDF's joint chain, not the base placement).

STAGE 2: full world-frame comparison, only run if stage 1 passes. Adds the
base's world transform (position AND rotation, applied as rot @ local + pos)
on top, across a spread of live sim configurations (not just q=0).
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import numpy as np
import torch
import pytorch_kinematics as pk

from favor_fault_runner import FaultRobomimicImageRunner

URDF = "/workspace/RoboTwin/envs/curobo/src/curobo/content/assets/robot/franka_description/franka_panda.urdf"
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
print("chain joints:", chain.get_joint_parameter_names())

runner = FaultRobomimicImageRunner(
    output_dir="/tmp/verify_fk_final", dataset_path=DATASET, shape_meta=SHAPE_META,
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
print("robot0_base world pos:", np.round(base_pos, 4), " rot==I:", np.allclose(base_rot, np.eye(3)))

def get_state():
    q = np.array([raw.sim.data.qpos[a] for a in addrs])
    sim_pos = np.array(raw.sim.data.get_body_xpos("robot0_right_hand"))
    sim_rot = np.array(raw.sim.data.get_body_xmat("robot0_right_hand")).reshape(3, 3)
    return q, sim_pos, sim_rot

print()
print("=== STAGE 1: local-frame check (base offset subtracted out) ===")
q, sim_pos, sim_rot = get_state()
sim_pos_local = base_rot.T @ (sim_pos - base_pos)  # world -> base-local frame
q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
m = chain.forward_kinematics(q_t).get_matrix()
fk_pos_local = m[0, :3, 3].numpy()
fk_rot_local = m[0, :3, :3].numpy()
print("q                :", np.round(q, 4))
print("sim EE, base-local frame:", np.round(sim_pos_local, 4))
print("FK  EE, URDF-root frame :", np.round(fk_pos_local, 4))
stage1_err = np.linalg.norm(sim_pos_local - fk_pos_local)
print("STAGE 1 position error:", round(stage1_err, 5))
print("STAGE 1:", "PASS" if stage1_err < 0.01 else "FAIL -- URDF root and MuJoCo base frame are not simply related; stopping before stage 2.")

if stage1_err >= 0.01:
    sys.exit(0)

print()
print("=== STAGE 2: full world-frame check across 20 live configurations ===")
errs_pos, errs_rot = [], []
for trial in range(20):
    for _ in range(5):
        env.step(np.random.uniform(-0.3, 0.3, size=(1, 8)).astype(np.float32))
    q, sim_pos, sim_rot = get_state()
    q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    m = chain.forward_kinematics(q_t).get_matrix()
    fk_pos_local = m[0, :3, 3].numpy()
    fk_rot_local = m[0, :3, :3].numpy()
    fk_pos_world = base_rot @ fk_pos_local + base_pos
    fk_rot_world = base_rot @ fk_rot_local
    errs_pos.append(np.linalg.norm(fk_pos_world - sim_pos))
    errs_rot.append(np.linalg.norm(fk_rot_world - sim_rot))

errs_pos = np.array(errs_pos)
errs_rot = np.array(errs_rot)
print("position error (m):  mean=%.6f  max=%.6f" % (errs_pos.mean(), errs_pos.max()))
print("rotation error (Frobenius): mean=%.6f  max=%.6f" % (errs_rot.mean(), errs_rot.max()))
print()
print("D1 VERDICT:", "PASS" if errs_pos.max() < 0.01 else "FAIL")
