"""
FaultRobomimicImageRunner — subclass of the official RobomimicImageRunner.

Only __init__ is overridden, and only to insert FaultInjector between the
raw robomimic env and RobomimicImageWrapper (env_fn is an inline closure in
the official class, so there is no smaller hook point available without
editing the official file itself, which we do not do).

run() is NOT overridden — inherited byte-for-byte from RobomimicImageRunner,
so the policy rollout / video / logging logic is identical to Phase 2.
"""
import os, collections, pathlib, math, dill
import h5py
import wandb.sdk.data_types.video as wv
import sys
sys.path.insert(0, "/workspace/diffusion_policy")

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner, create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils

from fault_injector import FaultInjector

# RobomimicImageWrapper is a plain gym.Env (not gym.Wrapper), so it does NOT
# forward unknown attribute lookups to self.env the way gym.Wrapper does.
# This breaks the call chain AsyncVectorEnv.call('get_fault_info') needs to
# reach FaultInjector, which sits one level further in (env.env.env from the
# MultiStepWrapper's perspective). We patch this in at the CLASS level here
# (never touching the official file on disk) -- purely additive: any existing
# attribute lookup that already succeeds is unaffected, this only catches
# names that would otherwise raise AttributeError.
if not getattr(RobomimicImageWrapper, "_favor_getattr_patched", False):
    def _favor_forward_getattr(self, name):
        return getattr(self.env, name)
    RobomimicImageWrapper.__getattr__ = _favor_forward_getattr
    RobomimicImageWrapper._favor_getattr_patched = True


class FaultRobomimicImageRunner(RobomimicImageRunner):
    def __init__(self, output_dir, dataset_path, shape_meta,
            fault_joint_name, fault_type, fault_severity,
            n_train=0, n_train_vis=0, train_start_idx=0,
            n_test=5, n_test_vis=0, test_start_seed=10000,
            max_steps=400, n_obs_steps=2, n_action_steps=8,
            render_obs_key='agentview_image', fps=10, crf=22,
            past_action=False, abs_action=True, tqdm_interval_sec=5.0,
            n_envs=None):
        # Deliberately NOT calling super().__init__() — it builds env_fn
        # internally with no hook point. Instead we re-run the same
        # construction logic here, with one line changed (FaultInjector
        # inserted). Everything else (AsyncVectorEnv, MultiStepWrapper,
        # VideoRecordingWrapper, init_fn logic, run()) is identical to
        # the official class and run() is inherited unchanged below.
        from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
        BaseImageRunner.__init__(self, output_dir)

        if n_envs is None:
            n_envs = n_train + n_test
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def make_wrapped(enable_render):
            robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta, enable_render=enable_render)
            robomimic_env.env.hard_reset = False
            faulted = FaultInjector(robomimic_env, fault_joint_name, fault_type, fault_severity)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=faulted,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec='h264', input_pix_fmt='rgb24',
                        crf=crf, thread_type='FRAME', thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        def env_fn():
            return make_wrapped(enable_render=True)

        def dummy_env_fn():
            return make_wrapped(enable_render=False)

        env_fns = [env_fn] * n_envs
        env_seeds, env_prefixs, env_init_fn_dills = [], [], []

        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]
                def init_fn(env, init_state=init_state, enable_render=enable_render, output_dir=output_dir):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath('media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state
                env_seeds.append(train_idx); env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis
            def init_fn(env, seed=seed, enable_render=enable_render, output_dir=output_dir):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath('media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)
            env_seeds.append(seed); env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
    # run() inherited unchanged from RobomimicImageRunner
