import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import numpy as np
from ik_projector import _project_waypoints_impl, solve_batch_ik
import ik_projector
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_locked_conv")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)
env_meta['env_kwargs']['use_object_obs'] = False
env_meta['env_kwargs']['controller_configs']['control_delta'] = False
robomimic_env = create_env(env_meta=env_meta, shape_meta=cfg.task.shape_meta, enable_render=True)
wrapped = RobomimicImageWrapper(env=robomimic_env, shape_meta=cfg.task.shape_meta, render_obs_key='agentview_image')

# reset until the env's OWN joint4 lock candidate (its current qpos at this
# reset) is captured, so we test with a REAL onset value, not a hardcoded one.
raw_obs = wrapped.reset()
q_lock_real = wrapped.env.env.sim.data.qpos[wrapped.env.env.sim.model.get_joint_qpos_addr("robot0_joint4")]
print("using REAL q_lock for joint4 from this reset:", q_lock_real)

n_obs_steps = cfg.task.env_runner.n_obs_steps
obs_dict = {}
for key, val in raw_obs.items():
    v = torch.from_numpy(np.asarray(val)).float().unsqueeze(0)
    obs_dict[key] = v.unsqueeze(1).repeat(1, n_obs_steps, *([1] * (v.dim() - 1))).to("cuda:0")

with torch.no_grad():
    out = base_policy.predict_action(obs_dict)
raw_clean = out['action_pred'][:, :16, :]

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
fault_spec_locked = {'joint_idx': 3, 'q_lock': float(q_lock_real), 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}
q_current = wrapped.env.env.sim.data.qpos[[wrapped.env.env.sim.model.jnt_qposadr[wrapped.env.env.sim.model.joint_name2id(f"robot0_joint{i}")] for i in range(1,8)]]
q_seed = torch.tensor(q_current, dtype=torch.float32).unsqueeze(0)
print("using REAL current qpos as q_ref seed:", q_seed)

errs = []
orig = ik_projector.solve_batch_ik
def traced(*a, **kw):
    q_sol, pos_err, rot_err = orig(*a, **kw)
    errs.append((pos_err.min().item(), rot_err.min().item()))
    return q_sol, pos_err, rot_err
ik_projector.solve_batch_ik = traced
_project_waypoints_impl(raw_clean, fault_spec_locked, q_seed, K=64, iters=5, lambda_reg=0.5)
ik_projector.solve_batch_ik = orig

print("=== LOCKED (real q_lock) IK convergence per waypoint ===")
for i, (p, r) in enumerate(errs):
    print(f"  waypoint {i}: pos_err_min={p:.4f}  rot_err_min={r:.4f}")
print("max over waypoints: pos_err_min=", max(e[0] for e in errs), " rot_err_min=", max(e[1] for e in errs))

# --- blend_frac schedule across all 100 denoising steps ---
scheduler = base_policy.noise_scheduler
scheduler.set_timesteps(base_policy.num_inference_steps)
print("=== blend_frac = sqrt(alpha_prod_t) across denoising steps ===")
fracs = []
for t in scheduler.timesteps:
    alpha_prod_t = scheduler.alphas_cumprod[t]
    blend_frac = torch.sqrt(alpha_prod_t).item()
    fracs.append(blend_frac)
print("first 10 steps (most noise):", [f"{f:.4f}" for f in fracs[:10]])
print("last 10 steps (least noise):", [f"{f:.4f}" for f in fracs[-10:]])
print("num steps with blend_frac > 0.5:", sum(1 for f in fracs if f > 0.5))
print("num steps with blend_frac > 0.9:", sum(1 for f in fracs if f > 0.9))
