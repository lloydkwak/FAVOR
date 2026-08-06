"""
Checks whether the C-step's (mid-denoising) ProjectWaypoints suffers the
SAME lambda_reg convergence-blocking bug just found and fixed in the
terminal joint-target IK. Uses the same methodology: thorough unconstrained
search (K=500, iters=50, lambda_reg=0) as ground truth, vs the C-step's
actual default (lambda_reg=0.5), on a real target from an actual policy
rollout under a locked-joint fault (the C-step's actual use case).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import numpy as np
from ik_projector import solve_batch_ik, _rot6d_to_matrix
from panda_kinematics import panda_fk
from favor_policy import FavorHybridImagePolicy
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_cstep_diag")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

# OSC mode, joint3 locked -- the C-step's actual real-world use case
runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_cstep_diag_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type="locked", fault_severity=None,
    n_train=0, n_test=1, test_start_seed=10000, n_envs=1,
    max_steps=8,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
    actuation_mode='osc',
)
fault_spec = {'joint_idx': 2, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}
favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, env_ref=runner.env)

obs = runner.env.reset()
favor.reset()
device = base_policy.device
q_lock_val = runner.env.call('get_fault_info')[0]['q_lock']
fault_spec['q_lock'] = q_lock_val
print("actual q_lock:", q_lock_val)

q_current = np.array(runner.env.call('get_current_qpos')[0])
q_current[2] = q_lock_val  # matches how C-step seeds
print("q_current (joint3 forced to lock):", q_current)

obs_dict = {k: torch.from_numpy(v).to(device=device) for k, v in dict(obs).items()}
with torch.no_grad():
    action_dict = favor.predict_action(obs_dict)
action_pred = action_dict['action_pred'][0].detach().cpu()  # (16,10)

target_pos = action_pred[0, 0:3]
target_rot = _rot6d_to_matrix(action_pred[0, 3:9].unsqueeze(0)).squeeze(0)
print("C-step's typical target pos:", target_pos.numpy())

locked_mask = torch.zeros(7, dtype=torch.bool); locked_mask[2] = True
K = 500
seeds = torch.rand(K, 7) * (q_hi - q_lo) + q_lo
seeds[:, 2] = q_lock_val
q_lock_vec = torch.zeros(K, 7); q_lock_vec[:, 2] = q_lock_val
target_pos_k = target_pos.unsqueeze(0).expand(K, -1)
target_rot_k = target_rot.unsqueeze(0).expand(K, -1, -1)
q_ref_k = torch.tensor(q_current, dtype=torch.float32).unsqueeze(0).expand(K, -1)

for lam in [0.0, 0.05, 0.5]:
    q_sol, pos_err, rot_err = solve_batch_ik(
        target_pos_k, target_rot_k, seeds, q_lo, q_hi, locked_mask, q_lock_vec, q_ref_k,
        iters=50, lambda_reg=lam)
    print(f"lambda_reg={lam}: best pos_err={pos_err.min().item()*1000:.2f}mm, "
          f"converged(<10mm)={100*(pos_err<0.01).float().mean().item():.1f}%")
