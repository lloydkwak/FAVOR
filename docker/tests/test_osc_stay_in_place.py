import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from scipy.spatial.transform import Rotation

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
wrapped = RobomimicImageWrapper(env=robomimic_env, shape_meta=shape_meta, render_obs_key='agentview_image')
obs = wrapped.reset()

target_pos = obs['robot0_eef_pos'].copy()
target_quat = obs['robot0_eef_quat'].copy()
target_axis_angle = Rotation.from_quat(target_quat).as_rotvec()
print("initial pos:", target_pos, " initial axis_angle:", target_axis_angle)

for step in range(10):
    action = np.concatenate([target_pos, target_axis_angle, [-1.0]]).astype(np.float32)
    obs, reward, done, info = wrapped.step(action)
    achieved_pos = obs['robot0_eef_pos'].copy()
    achieved_quat = obs['robot0_eef_quat'].copy()
    pos_gap = np.linalg.norm(achieved_pos - target_pos)
    rot_gap = np.linalg.norm(Rotation.from_quat(achieved_quat).as_rotvec() - target_axis_angle)
    print(f"step {step}: pos_gap={pos_gap*1000:.2f}mm  rot_gap={rot_gap:.4f}rad  achieved_pos={achieved_pos}")
