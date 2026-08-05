"""
Regression test for the KEY finding: under joint-lock fault, OSC_POSE's
Jacobian-based control produces genuine closed-loop instability on axes
that were never commanded to move -- not just "undershoot" from a missing
DOF. Same seed, same naive (uncorrected) target script executed in a
fault-active vs fault-free env; the Y-axis divergence must exceed a large
threshold and its sign must flip mid-trajectory (oscillation), confirming
this is dynamical instability, not a simple offset.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from fault_injector import FaultInjector
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

def make_env(with_fault):
    env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)
    env_meta['env_kwargs']['use_object_obs'] = False
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta, enable_render=True)
    env = FaultInjector(robomimic_env, "robot0_joint3", "locked", None) if with_fault else robomimic_env
    return RobomimicImageWrapper(env=env, shape_meta=shape_meta, render_obs_key='agentview_image')

env_a = make_env(with_fault=True)
env_b = make_env(with_fault=False)
env_a.seed(10000); obs_a = env_a.reset()
env_b.seed(10000); obs_b = env_b.reset()

start_pos = obs_a['robot0_eef_pos'].copy()
axis_angle = Rotation.from_quat(obs_a['robot0_eef_quat'].copy()).as_rotvec()

y_diffs_mm = []
for step in range(10):
    frac = (step + 1) / 10
    target_pos = start_pos + np.array([0.05 * frac, 0.0, 0.0])
    action = np.concatenate([target_pos, axis_angle, [-1.0]]).astype(np.float32)
    obs_a, _, _, _ = env_a.step(action)
    obs_b, _, _, _ = env_b.step(action)
    y_diffs_mm.append((obs_a['robot0_eef_pos'][1] - obs_b['robot0_eef_pos'][1]) * 1000)

print("Y-axis divergence (fault - no_fault) per step, mm:", [f"{v:.2f}" for v in y_diffs_mm])
max_abs = max(abs(v) for v in y_diffs_mm)
sign_flip = any(y_diffs_mm[i] * y_diffs_mm[i+1] < 0 for i in range(len(y_diffs_mm)-1))
print(f"max |divergence| = {max_abs:.2f} mm, sign flip observed = {sign_flip}")

assert max_abs > 30, f"expected large (>30mm) Y-divergence under fault, got {max_abs:.2f}mm -- instability may have been masked/fixed elsewhere"
assert sign_flip, "expected a sign flip (oscillation), got monotonic drift only -- re-check whether this is still genuine instability"
print("LOCKED_JOINT_OSC_INSTABILITY_CONFIRMED")
