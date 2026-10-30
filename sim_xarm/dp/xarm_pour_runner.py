"""Env runners for the sim_xarm pouring task inside diffusion_policy training."""
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NullRunner(BaseImageRunner):
    """No rollouts during training (checkpoint on train_loss, e.g. checkpoint=joint_trainloss_topk)."""

    def __init__(self, output_dir=None, **kwargs):
        super().__init__(output_dir)

    def run(self, policy):
        return {}
