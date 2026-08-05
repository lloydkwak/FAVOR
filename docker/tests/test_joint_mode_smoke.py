"""
First end-to-end smoke test of actuation_mode='joint' running through the
FULL FaultRobomimicImageRunner stack (AsyncVectorEnv -> MultiStepWrapper ->
VideoRecordingWrapper -> RobomimicImageWrapper -> FaultInjector ->
JointActuationWrapper -> raw robosuite env), inside a forked worker process
(not the main process, unlike all prior unit tests). No policy involved --
manually commands "stay at current joint config" for a few steps via
env.call_each, purely to confirm the wrapper chain doesn't crash under
AsyncVectorEnv's fork/pipe machinery.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
from favor_fault_runner import FaultRobomimicImageRunner

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

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_scratch_joint_smoke",
    dataset_path=DATASET, shape_meta=shape_meta,
    fault_joint_name="robot0_joint3", fault_type=None, fault_severity=None,
    n_train=0, n_test=1, n_envs=1,
    max_steps=10, n_obs_steps=2, n_action_steps=8,
    render_obs_key='agentview_image', abs_action=True,
    actuation_mode='joint',
)
print("runner constructed OK, actuation_mode =", runner.actuation_mode, " self.abs_action =", runner.abs_action)

env = runner.env
obs = env.reset()
print("reset OK, obs keys:", list(obs.keys()))
print("robot0_eef_pos shape:", obs['robot0_eef_pos'].shape)

import numpy as np
qpos_list = env.call('get_current_qpos')
print("get_current_qpos via RPC:", qpos_list)

# Build a "stay put" 8-step chunk: target = current qpos, gripper open (-1)
q_now = np.array(qpos_list[0])
action_chunk = np.tile(np.concatenate([q_now, [-1.0]]), (1, 8, 1)).astype(np.float32)  # (n_envs=1, n_action_steps=8, 8)
obs, reward, done, info = env.step(action_chunk)
print("step OK, obs keys:", list(obs.keys()), " reward:", reward, " done:", done)
print("SMOKE_TEST_PASSED")
