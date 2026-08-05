import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import numpy as np
from ik_projector import _project_waypoints_impl
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_ikdiag2")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

# --- build a REAL Lift env and get a REAL observation ---
env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)
env_meta['env_kwargs']['use_object_obs'] = False
env_meta['env_kwargs']['controller_configs']['control_delta'] = False
robomimic_env = create_env(env_meta=env_meta, shape_meta=cfg.task.shape_meta, enable_render=True)
wrapped = RobomimicImageWrapper(env=robomimic_env, shape_meta=cfg.task.shape_meta, render_obs_key='agentview_image')
raw_obs = wrapped.reset()

n_obs_steps = cfg.task.env_runner.n_obs_steps
obs_dict = {}
for key, val in raw_obs.items():
    v = torch.from_numpy(np.asarray(val)).float().unsqueeze(0)  # (1, ...)
    obs_dict[key] = v.unsqueeze(1).repeat(1, n_obs_steps, *([1] * (v.dim() - 1))).to("cuda:0")

with torch.no_grad():
    out = base_policy.predict_action(obs_dict)
raw_clean = out['action_pred'][:, :16, :]
print("raw_clean action range (pos cols 0:3):", raw_clean[..., 0:3].min().item(), raw_clean[..., 0:3].max().item())
print("raw eef_pos from real obs (for comparison):", raw_obs.get('robot0_eef_pos'))

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
fault_spec = {'joint_idx': 3, 'q_lock': -2.6, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}
q_seed = torch.zeros(1, 7)

import ik_projector
orig_solve = ik_projector.solve_batch_ik
def traced_solve(*args, **kwargs):
    q_sol, pos_err, rot_err = orig_solve(*args, **kwargs)
    print(f"  IK batch: pos_err min/mean/max = {pos_err.min().item():.4f}/{pos_err.mean().item():.4f}/{pos_err.max().item():.4f}"
          f"   rot_err min/mean/max = {rot_err.min().item():.4f}/{rot_err.mean().item():.4f}/{rot_err.max().item():.4f}")
    return q_sol, pos_err, rot_err
ik_projector.solve_batch_ik = traced_solve

corrected, last_seed = _project_waypoints_impl(raw_clean, fault_spec, q_seed, K=64, iters=5)
print("done")

# --- sanity: also solve WITHOUT the lock, using the SAME raw_clean targets,
#     to confirm the targets themselves are reachable in principle ---
print("=== sanity check: same targets, NO fault ===")
fault_spec_none_like = {'joint_idx': 3, 'q_lock': -2.6, 'fault_type': None, 'q_lo': q_lo, 'q_hi': q_hi}
_project_waypoints_impl(raw_clean, fault_spec_none_like, q_seed, K=64, iters=5)
