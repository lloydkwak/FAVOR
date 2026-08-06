"""
Compares descent speed (z-height over the same number of macro-steps) between
actuation_mode='osc' and 'joint', same seed, no fault, to see whether JOINT
mode is simply progressing toward the grasp much more slowly (explaining why
gripper hasn't closed within 80 steps) or is stuck/off-target entirely.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_speed_cmp")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

def run(actuation_mode, max_steps):
    runner = FaultRobomimicImageRunner(
        output_dir=f"/workspace/results/_speed_{actuation_mode}",
        dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
        fault_joint_name="robot0_joint3", fault_type=None, fault_severity=None,
        n_train=0, n_test=1, test_start_seed=10000, n_envs=1,
        max_steps=max_steps,
        n_obs_steps=cfg.task.env_runner.n_obs_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        render_obs_key=cfg.task.env_runner.render_obs_key,
        abs_action=cfg.task.env_runner.abs_action,
        actuation_mode=actuation_mode,
    )
    if actuation_mode == 'joint':
        favor = FavorHybridImagePolicy(base_policy, fault_spec=None, env_ref=runner.env,
                                        actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)
    else:
        favor = base_policy

    obs = runner.env.reset()
    policy = favor
    policy.reset()
    device = base_policy.device
    done = False
    step_count = 0
    z_trace = []
    while not done and step_count < max_steps:
        obs_dict = {k: torch.from_numpy(v).to(device=device) for k, v in dict(obs).items()}
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        action = action_dict['action'].detach().to('cpu').numpy()
        if actuation_mode == 'osc':
            # mirror what RobomimicImageRunner.run() does before env.step()
            # when self.abs_action=True: rotation_6d(6) -> axis_angle(3) via
            # the SAME RotationTransformer runner.rotation_transformer already
            # built (not a new one -- avoids any convention mismatch).
            raw_shape = action.shape
            pos = action[..., :3]
            rot6d = action[..., 3:9]
            gripper = action[..., [-1]]
            axis_angle = runner.rotation_transformer.inverse(rot6d)
            env_action = np.concatenate([pos, axis_angle, gripper], axis=-1)
        else:
            env_action = action
        obs, reward, done, info = runner.env.step(env_action)
        z_trace.append(obs['robot0_eef_pos'][0, -1, 2])
        done = np.all(done)
        step_count += action.shape[1]
    return np.array(z_trace)

print("=== OSC mode, 80 steps ===")
z_osc = run('osc', 80)
print("z trace:", z_osc)

print("\n=== JOINT mode, full max_steps (400) ===")
z_joint = run('joint', 400)
print("z trace (every 5th value):", z_joint[::5])
print("min z reached:", z_joint.min(), " at step", z_joint.argmin()*8)
