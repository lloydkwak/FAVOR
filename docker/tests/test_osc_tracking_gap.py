"""
Measures the gap between COMMANDED EE-pose (what we send as an abs_action)
and ACHIEVED EE-pose (what the robot's gripper actually reaches after the
OSC_POSE controller + physics step), under the joint4-locked fault vs
no fault. This tests the previously-unverified assumption that "an
IK-feasible corrected pose target will actually be tracked" -- robosuite's
OSC_POSE uses a nullspace controller that has NO knowledge of the fault,
so a large gap here would explain why IK-quality improvements alone
haven't moved success rates.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from fault_injector import FaultInjector
from panda_kinematics import panda_fk
from ik_projector import _rot6d_to_matrix, _matrix_to_rot6d

DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"

def get_shape_meta():
    return {
        'obs': {
            'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
            'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
            'robot0_eef_pos': {'shape': [3]},
            'robot0_eef_quat': {'shape': [4]},
            'robot0_gripper_qpos': {'shape': [2]},
        },
        'action': {'shape': [10]},
    }

def run_trial(use_fault):
    env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)
    env_meta['env_kwargs']['use_object_obs'] = False
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    robomimic_env = create_env(env_meta=env_meta, shape_meta=get_shape_meta(), enable_render=True)

    if use_fault:
        env = FaultInjector(robomimic_env, "robot0_joint4", "locked", None)
    else:
        env = robomimic_env

    wrapped = RobomimicImageWrapper(env=env, shape_meta=get_shape_meta(), render_obs_key='agentview_image')
    obs = wrapped.reset()

    sim = robomimic_env.env.sim if not use_fault else robomimic_env.env.sim
    q_current = np.array([sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint_name2id(f"robot0_joint{i}")]] for i in range(1,8)])
    q_lock = q_current[3] if use_fault else None

    current_pos = obs['robot0_eef_pos'].copy()
    current_quat = obs['robot0_eef_quat'].copy()  # xyzw

    # Build a small, IK-feasible target: move 3cm in +x, keep orientation.
    # If use_fault, verify feasibility via our own IK first (should be easy,
    # small motion, well within reach even with joint4 fixed).
    target_pos = current_pos + np.array([0.03, 0.0, 0.0])

    gaps = []
    for step in range(10):
        # abs_action format: pos(3) + axis_angle(3) + gripper(1) = 7-dim raw action
        # (robosuite's native abs action, NOT the 10-dim rotation_6d used by the policy)
        from scipy.spatial.transform import Rotation
        quat_xyzw = current_quat
        axis_angle = Rotation.from_quat(quat_xyzw).as_rotvec()
        action = np.concatenate([target_pos, axis_angle, [-1.0]]).astype(np.float32)  # gripper open

        obs, reward, done, info = wrapped.step(action)
        achieved_pos = obs['robot0_eef_pos'].copy()
        gap = np.linalg.norm(achieved_pos - target_pos)
        gaps.append(gap)
        current_quat = obs['robot0_eef_quat'].copy()

    return gaps, q_lock

print("=== NO FAULT ===")
gaps_nofault, _ = run_trial(use_fault=False)
for i, g in enumerate(gaps_nofault):
    print(f"  step {i}: |achieved - commanded| = {g*1000:.2f} mm")

print("=== FAULT (joint4 locked) ===")
gaps_fault, q_lock = run_trial(use_fault=True)
print(f"  (locked joint4 at {q_lock:.4f})")
for i, g in enumerate(gaps_fault):
    print(f"  step {i}: |achieved - commanded| = {g*1000:.2f} mm")

print("=" * 60)
print(f"final gap, no fault:  {gaps_nofault[-1]*1000:.2f} mm")
print(f"final gap, fault:     {gaps_fault[-1]*1000:.2f} mm")
