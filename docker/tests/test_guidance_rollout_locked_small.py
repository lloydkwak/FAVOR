"""
Second rollout connection test: joint3 locked, EmbodimentGuidance active.
n=5 first-look, not yet a paired statistical comparison -- just checking (a)
it runs without crashing/NaN, (b) score is non-trivially different from 0
(the floor B1/naive-projection results kept hitting).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from embodiment_guidance import EmbodimentGuidance
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_guidance_locked")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_guidance_locked_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type="locked", fault_severity=None,
    n_train=0, n_test=5, test_start_seed=10000, n_envs=5,
    max_steps=cfg.task.env_runner.max_steps,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
    actuation_mode='osc',
)
fault_spec = {'joint_idx': 2, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}
guidance = EmbodimentGuidance(delta_max=0.2, ik_iters=3)
favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, env_ref=runner.env, guidance=guidance)
log = runner.run(favor)
print("EmbodimentGuidance, joint3 locked, Lift, n=5 -> test/mean_score =", log.get("test/mean_score"))
print("(B1 reference, same condition: 0.39 with n=28)")
