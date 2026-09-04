"""
No-op env runner for training-only runs.

train_diffusion_unet_hybrid_workspace.py unconditionally instantiates
cfg.task.env_runner (line ~108) and asserts it is a BaseImageRunner, even
when rollouts are disabled -- so a training run cannot simply omit the
runner from the config. For LIBERO that instantiation fails outright,
because FaultRobomimicImageRunner reconstructs the environment from the
dataset's env_args, and LIBERO's env_name ("Libero_Floor_Manipulation")
is registered by the `libero` package, which is not installed in this
robodiff image (robosuite only knows Lift, Stack, PickPlace, ... ).

This class satisfies the instantiation and the isinstance assert while
doing nothing, so joint-space training on LIBERO data can proceed before
the evaluation path (LIBERO env registration + JOINT_POSITION controller
+ FaultInjector wiring) is in place. run() returns an empty dict, which is
what the training loop merges into its step log; it is never called while
training.rollout_every is set beyond num_epochs.

Use ONLY for training. Any run that reports test_mean_score must use the
real runner -- with this one, checkpoint.topk.monitor_key would never be
populated and topk selection would be meaningless.
"""
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NullEnvRunner(BaseImageRunner):
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(output_dir)

    def run(self, policy):
        return dict()
