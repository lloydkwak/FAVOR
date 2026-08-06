"""
NullEnvRunner — a no-op BaseImageRunner, used ONLY during joint-action
training (lift_image_joint.yaml) to disable the official
train_diffusion_unet_hybrid_workspace.py's periodic in-training rollout
(cfg.training.rollout_every).

Why this exists: the official env_runner (RobomimicImageRunner) assumes
EE-pose(10-dim: pos+rot6d+gripper) actions and calls undo_transform_action
(rot6d -> axis_angle) on whatever the policy predicts. Our joint-action
policy predicts 8-dim (joint_pos(7)+gripper(1)) actions, which would crash
that conversion. Rather than relying on rollout_every's timing (epoch 0
always triggers rollout regardless of rollout_every's value, since
0 % anything == 0), we replace env_runner entirely with this safe no-op --
zero risk of the crash regardless of epoch count.

Real evaluation of the joint-action checkpoint happens SEPARATELY, after
training, via FaultRobomimicImageRunner(actuation_mode='joint') (already
built and validated this session) -- not via this in-training runner.
"""
import sys, os
sys.path.insert(0, "/workspace/diffusion_policy")
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NullEnvRunner(BaseImageRunner):
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(output_dir)

    def run(self, policy):
        return {}
