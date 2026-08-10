"""
NullEnvRunner — a no-op BaseImageRunner, used ONLY during joint-action
training to disable the official train_diffusion_unet_hybrid_workspace.py's
periodic in-training rollout (cfg.training.rollout_every). The official
env_runner (RobomimicImageRunner) assumes EE-pose(10-dim) actions and would
crash on our 8-dim joint actions. Real evaluation happens separately, via
FaultRobomimicImageRunner(actuation_mode='joint').
"""
import sys, os
sys.path.insert(0, "/workspace/diffusion_policy")
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NullEnvRunner(BaseImageRunner):
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(output_dir)

    def run(self, policy):
        return {}
