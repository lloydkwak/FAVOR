"""
Large-scale (n=28, matching OSC B1's original evaluation scale) confirmation
of actuation_mode='joint' B1 (no fault) now that lambda_reg=0.0 fixed the
IK convergence bug. Prior results at this scale: OSC B1 = 1.00.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
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
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_b1_n28")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_b1_joint_n28_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type=None, fault_severity=None,
    n_train=0, n_test=28, test_start_seed=10000, n_envs=28,
    max_steps=cfg.task.env_runner.max_steps,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
    actuation_mode='joint',
)
favor = FavorHybridImagePolicy(base_policy, fault_spec=None, env_ref=runner.env,
                                actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)
log = runner.run(favor)
print("B1, JOINT mode, no fault, Lift, n=28 -> test/mean_score =", log.get("test/mean_score"))
print("(OSC B1 reference: 1.00)")
