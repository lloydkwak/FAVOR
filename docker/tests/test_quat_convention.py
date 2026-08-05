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

sim = robomimic_env.env.sim
quat_obs = obs['robot0_eef_quat']
print("robot0_eef_quat (from obs):", quat_obs)

# ground truth: query the actual gripper site orientation as a rotation matrix directly from MuJoCo
grip_site_id = sim.model.site_name2id("gripper0_grip_site")
mat_from_mujoco = sim.data.site_xmat[grip_site_id].reshape(3,3)
print("rotation matrix directly from MuJoCo site_xmat:\n", mat_from_mujoco)

# hypothesis A: quat_obs is [x,y,z,w] (scipy default)
rot_A = Rotation.from_quat(quat_obs)
print("\nHypothesis A (xyzw): rotation matrix:\n", rot_A.as_matrix())
print("Hypothesis A diff from MuJoCo ground truth (Frobenius norm):", np.linalg.norm(rot_A.as_matrix() - mat_from_mujoco))

# hypothesis B: quat_obs is [w,x,y,z] (MuJoCo native) -- reorder to [x,y,z,w] for scipy
quat_reordered = np.array([quat_obs[1], quat_obs[2], quat_obs[3], quat_obs[0]])
rot_B = Rotation.from_quat(quat_reordered)
print("\nHypothesis B (wxyz->reordered): rotation matrix:\n", rot_B.as_matrix())
print("Hypothesis B diff from MuJoCo ground truth (Frobenius norm):", np.linalg.norm(rot_B.as_matrix() - mat_from_mujoco))
