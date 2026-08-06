"""
Directly tests the user's dilemma: does OSC_POSE's fault-blindness cause
divergence even when the target EE-pose is GUARANTEED reachable with the
locked joint fixed (i.e., the target itself was generated via FK from a
config with joint3 already at its locked value)? This isolates "OSC doesn't
know about the fault" from "the target itself was infeasible" -- our earlier
Y-axis divergence test (60.8mm, sign flip) used B1's NAIVE uncorrected
target, which was never confirmed reachable under the lock.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from fault_injector import FaultInjector
from panda_kinematics import panda_fk

DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
shape_meta = {
    'obs': {
        'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'robot0_eef_pos': {'shape': [3]},
        'robot0_eef_quat': {'shape': [4]},
        'robot0_gripper_qpos': {'shape': [2]},
    },
    'action': {'shape': [10]},
}
env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)
env_meta['env_kwargs']['use_object_obs'] = False
env_meta['env_kwargs']['controller_configs']['control_delta'] = False
robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta, enable_render=True)
env = FaultInjector(robomimic_env, "robot0_joint3", "locked", None)
wrapped = RobomimicImageWrapper(env=env, shape_meta=shape_meta, render_obs_key='agentview_image')

wrapped.seed(10000)
obs = wrapped.reset()
sim = env._sim()
qpos_addrs = [sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1, 8)]
q_current = np.array([sim.data.qpos[a] for a in qpos_addrs])
q_lock = q_current[2]
print("locked value (joint3):", q_lock)

# Build a GUARANTEED-REACHABLE target: take the current config, move ONLY
# the free joints by a modest amount, keep joint3 exactly at its lock value,
# then use FK to get the corresponding EE pose. This target is reachable
# by construction -- if OSC still diverges, that's a clean, unconfounded
# demonstration of "OSC's fault-blindness alone causes failure."
q_target = q_current.copy()
q_target[0] += 0.3   # joint1
q_target[1] += 0.2   # joint2
q_target[3] += 0.2   # joint4 (small nudge, stay within limits)
q_target[2] = q_lock  # joint3 stays exactly at its locked value
q_target_t = torch.tensor(q_target, dtype=torch.float32).unsqueeze(0)
target_pos, target_rot = panda_fk(q_target_t)
target_pos = target_pos.squeeze(0).numpy()
target_axis_angle = Rotation.from_matrix(target_rot.squeeze(0).numpy()).as_rotvec()
print("reachable target EE pos:", target_pos)

y0 = obs['robot0_eef_pos'][1]
print(f"\n{'step':>4} {'ach_x':>9} {'ach_y':>9} {'ach_z':>9} {'gap_to_target_mm':>18}")
for step in range(30):
    action = np.concatenate([target_pos, target_axis_angle, [-1.0]]).astype(np.float32)
    obs, reward, done, info = wrapped.step(action)
    ach = obs['robot0_eef_pos']
    gap = np.linalg.norm(ach - target_pos) * 1000
    print(f"{step:>4} {ach[0]:>9.4f} {ach[1]:>9.4f} {ach[2]:>9.4f} {gap:>18.2f}")
