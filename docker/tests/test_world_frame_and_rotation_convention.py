"""
Regression test for two compounding bugs found via manual IK convergence
diagnostics:
(1) rotation_6d must be ROW-based (matches pytorch3d, which diffusion_policy's
    RotationTransformer uses), not column-based.
(2) panda_fk/panda_jacobian must return WORLD-frame coordinates (robot base
    offset baked in), since abs_action policy outputs are in world frame.
Both bugs independently caused IK to converge to nonsense (rot_err stuck near
pi, pos_err stuck ~0.4-0.6m) even with NO fault applied at all.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import numpy as np
from ik_projector import _project_waypoints_impl, solve_batch_ik
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_wf_test")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)
env_meta['env_kwargs']['use_object_obs'] = False
env_meta['env_kwargs']['controller_configs']['control_delta'] = False
robomimic_env = create_env(env_meta=env_meta, shape_meta=cfg.task.shape_meta, enable_render=True)
wrapped = RobomimicImageWrapper(env=robomimic_env, shape_meta=cfg.task.shape_meta, render_obs_key='agentview_image')
raw_obs = wrapped.reset()

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
fault_spec_none = {'joint_idx': 3, 'q_lock': 0.0, 'fault_type': None, 'q_lo': q_lo, 'q_hi': q_hi}
q_seed = torch.zeros(1, 7)

errs = []
orig = solve_batch_ik
def traced(*a, **kw):
    q_sol, pos_err, rot_err = orig(*a, **kw)
    errs.append((pos_err.min().item(), rot_err.min().item()))
    return q_sol, pos_err, rot_err
import ik_projector
ik_projector.solve_batch_ik = traced
_project_waypoints_impl(raw_clean, fault_spec_none, q_seed, K=64, iters=5)
ik_projector.solve_batch_ik = orig

min_pos_errs = [e[0] for e in errs]
min_rot_errs = [e[1] for e in errs]
print("min pos_err across waypoints:", min_pos_errs)
print("min rot_err across waypoints:", min_rot_errs)

assert max(min_pos_errs) < 0.01, f"no-fault IK failed to converge on position for some waypoint: {min_pos_errs}"
assert max(min_rot_errs) < 0.05, f"no-fault IK failed to converge on rotation for some waypoint: {min_rot_errs}"
print("WORLD_FRAME_AND_ROTATION_CONVENTION_VERIFIED")
