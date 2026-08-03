import sys, time
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from ik_projector import ProjectWaypoints

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_prof")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
fault_spec = {'joint_idx': 3, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}

shape_meta = cfg.task.shape_meta
B = 5
n_obs_steps = cfg.task.env_runner.n_obs_steps
obs_dict = {}
for key, attr in shape_meta['obs'].items():
    shape = tuple(attr['shape'])
    obs_dict[key] = torch.randn(B, n_obs_steps, *shape, device="cuda:0")

# --- (a) time with fault_spec=None (no projector call at all) ---
favor_none = FavorHybridImagePolicy(base_policy, fault_spec=None)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    favor_none.predict_action(obs_dict)
torch.cuda.synchronize()
t_none = time.time() - t0
print(f"predict_action, fault_spec=None (no IK at all): {t_none:.2f}s")

# --- (b) time with fault_spec set, projector active (full 100-step projection) ---
projector = ProjectWaypoints(K=64, iters=5)
favor_locked = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, projector=projector)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    favor_locked.predict_action(obs_dict)
torch.cuda.synchronize()
t_locked = time.time() - t0
print(f"predict_action, fault_spec=locked, ALL 100 steps projected: {t_locked:.2f}s")
print(f"overhead from projector: {t_locked - t_none:.2f}s over 100 denoising steps -> {(t_locked-t_none)/100*1000:.1f} ms/step")
print(f"-> per single IK waypoint solve (B*Tp={B*16} per step): {(t_locked-t_none)/100/(B*16)*1000:.2f} ms")
