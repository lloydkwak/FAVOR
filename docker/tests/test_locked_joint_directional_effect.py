"""
Isolates the PURE effect of the physical joint lock on achieved EE pose,
separate from generic receding-horizon tracking lag. Method: take the SAME
sequence of raw (uncorrected, un-FAVOR'd) target actions -- as if replaying
a fixed script -- and execute it in (a) a fault-active env (joint3 locked)
and (b) a fault-free env. If OSC's closed-loop feedback already "silently"
compensates for the missing DOF, achieved EE in (a) should differ from (b)
mainly in the direction joint3 would have moved the EE. If OSC does NOT
compensate well, the achieved-EE gap between (a) and (b) should be large and
systematic (not just generic noise).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from fault_injector import FaultInjector

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

def make_env(with_fault):
    env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)
    env_meta['env_kwargs']['use_object_obs'] = False
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta, enable_render=True)
    env = FaultInjector(robomimic_env, "robot0_joint3", "locked", None) if with_fault else robomimic_env
    return RobomimicImageWrapper(env=env, shape_meta=shape_meta, render_obs_key='agentview_image')

env_a = make_env(with_fault=True)
env_b = make_env(with_fault=False)

# Use the SAME seed so both envs start from an identical initial state.
env_a.seed(10000); obs_a = env_a.reset()
env_b.seed(10000); obs_b = env_b.reset()

# Script a fixed, UNCORRECTED target sequence: move 5cm in +x over 10 steps,
# same orientation held. This is deliberately naive (no IK correction at
# all) -- we want to see the raw physical effect of the lock alone.
from scipy.spatial.transform import Rotation
start_pos = obs_a['robot0_eef_pos'].copy()
start_quat = obs_a['robot0_eef_quat'].copy()
axis_angle = Rotation.from_quat(start_quat).as_rotvec()

print(f"{'step':>4} {'diff_x(mm)':>11} {'diff_y(mm)':>11} {'diff_z(mm)':>11} {'diff_norm(mm)':>14}")
for step in range(10):
    frac = (step + 1) / 10
    target_pos = start_pos + np.array([0.05 * frac, 0.0, 0.0])
    action = np.concatenate([target_pos, axis_angle, [-1.0]]).astype(np.float32)

    obs_a, _, _, _ = env_a.step(action)
    obs_b, _, _, _ = env_b.step(action)

    ach_a = obs_a['robot0_eef_pos']
    ach_b = obs_b['robot0_eef_pos']
    diff = (ach_a - ach_b) * 1000
    print(f"{step:>4} {diff[0]:>11.2f} {diff[1]:>11.2f} {diff[2]:>11.2f} {np.linalg.norm(diff):>14.2f}")

print()
print("final achieved pos, fault:   ", obs_a['robot0_eef_pos'])
print("final achieved pos, no fault:", obs_b['robot0_eef_pos'])
print("final divergence (mm):", np.linalg.norm(obs_a['robot0_eef_pos'] - obs_b['robot0_eef_pos'])*1000)
