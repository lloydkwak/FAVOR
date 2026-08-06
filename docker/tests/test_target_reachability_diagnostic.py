"""
Checks whether the policy's own predicted EE-pose target (the one that
produced 339.7mm IK residual even with K=64, iters=30) is even PHYSICALLY
reachable by the Panda arm at all -- independent of our IK solver's quality.
Does this by running a much more thorough, unconstrained search: 500 random
restarts, 50 iterations each, ZERO regularization (lambda_reg=0), to see
whether ANY solution gets close.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import numpy as np
from favor_policy import FavorHybridImagePolicy
from favor_fault_runner import FaultRobomimicImageRunner
from panda_kinematics import panda_fk
from ik_projector import solve_batch_ik, _rot6d_to_matrix

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_reach")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_reach_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type=None, fault_severity=None,
    n_train=0, n_test=1, test_start_seed=10000, n_envs=1,
    max_steps=8,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
    actuation_mode='joint',
)
favor = FavorHybridImagePolicy(base_policy, fault_spec=None, env_ref=runner.env,
                                actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)

obs = runner.env.reset()
favor.reset()
device = base_policy.device
q_current = np.array(runner.env.call('get_current_qpos')[0])
print("current qpos:", q_current)

obs_dict = {k: torch.from_numpy(v).to(device=device) for k, v in dict(obs).items()}
with torch.no_grad():
    action_dict = favor.predict_action(obs_dict)
action_pred = action_dict['action_pred'][0].detach().cpu()  # (16,10) raw EE-pose pred

target_pos = action_pred[0, 0:3]
target_rot = _rot6d_to_matrix(action_pred[0, 3:9].unsqueeze(0)).squeeze(0)
print("target pos:", target_pos.numpy())

pos0, rot0 = panda_fk(torch.tensor(q_current, dtype=torch.float32).unsqueeze(0))
print("current FK pos (should roughly match robot0_eef_pos obs):", pos0.squeeze(0).numpy())
print("distance from current EE to target:", (target_pos - pos0.squeeze(0)).norm().item() * 1000, "mm")

# Thorough, unconstrained (no fault, no regularization) search
K = 500
locked_mask = torch.zeros(7, dtype=torch.bool)
q_lock_vec = torch.zeros(K, 7)
seeds = torch.rand(K, 7) * (q_hi - q_lo) + q_lo
target_pos_k = target_pos.unsqueeze(0).expand(K, -1)
target_rot_k = target_rot.unsqueeze(0).expand(K, -1, -1)
q_ref_k = torch.tensor(q_current, dtype=torch.float32).unsqueeze(0).expand(K, -1)

q_sol, pos_err, rot_err = solve_batch_ik(
    target_pos_k, target_rot_k, seeds, q_lo, q_hi, locked_mask, q_lock_vec, q_ref_k,
    iters=50, lambda_reg=0.0)

print(f"\nThorough search (K=500, iters=50, lambda_reg=0):")
print(f"  best pos_err: {pos_err.min().item()*1000:.2f} mm")
print(f"  best rot_err: {rot_err[pos_err.argmin()].item()*180/3.14159:.2f} deg")
print(f"  fraction converged (pos_err<10mm): {(pos_err < 0.01).float().mean().item()*100:.1f}%")
print(f"  q_sol at best:", q_sol[pos_err.argmin()].numpy())
print(f"  q_lo:", q_lo.numpy())
print(f"  q_hi:", q_hi.numpy())

print("\n=== Comparison: same search WITH lambda_reg=0.3 (actual pipeline value) ===")
q_sol2, pos_err2, rot_err2 = solve_batch_ik(
    target_pos_k, target_rot_k, seeds, q_lo, q_hi, locked_mask, q_lock_vec, q_ref_k,
    iters=50, lambda_reg=0.3)
print(f"  best pos_err: {pos_err2.min().item()*1000:.2f} mm")
print(f"  fraction converged (pos_err<10mm): {(pos_err2 < 0.01).float().mean().item()*100:.1f}%")
