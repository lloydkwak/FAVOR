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

# check robot0_right_hand BODY orientation instead of grip_site
hand_mat = sim.data.get_body_xmat("robot0_right_hand")
print("robot0_right_hand body_xmat:\n", hand_mat)

quat_A = Rotation.from_quat(quat_obs)  # xyzw hypothesis
quat_reordered = np.array([quat_obs[1], quat_obs[2], quat_obs[3], quat_obs[0]])
quat_B = Rotation.from_quat(quat_reordered)  # wxyz hypothesis

print("\nHyp A (xyzw) vs robot0_right_hand diff:", np.linalg.norm(quat_A.as_matrix() - hand_mat))
print("Hyp B (wxyz) vs robot0_right_hand diff:", np.linalg.norm(quat_B.as_matrix() - hand_mat))

# also try the eef "reference" site robosuite actually uses internally
try:
    eef_site_id = sim.model.site_name2id("gripper0_ee")
    eef_mat = sim.data.site_xmat[eef_site_id].reshape(3,3)
    print("\ngripper0_ee site_xmat:\n", eef_mat)
    print("Hyp A vs gripper0_ee diff:", np.linalg.norm(quat_A.as_matrix() - eef_mat))
    print("Hyp B vs gripper0_ee diff:", np.linalg.norm(quat_B.as_matrix() - eef_mat))
except Exception as e:
    print("no gripper0_ee site:", e)
    print("available sites containing 'ee' or 'grip':", [n for n in sim.model.site_names if 'ee' in n.lower() or 'grip' in n.lower()])
