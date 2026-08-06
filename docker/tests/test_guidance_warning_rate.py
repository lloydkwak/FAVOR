"""
Quantifies how often EmbodimentGuidance's finite-check safety net fires
(vs how often guidance actually applies cleanly), after adding the
alpha_threshold early-step skip. Redirects the module's print-based warning
into a counter instead of scrolling raw logs.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import embodiment_guidance as eg

call_count = {"total": 0, "warned": 0, "skipped_low_alpha": 0}
orig_guide = eg.EmbodimentGuidance.guide
def counted_guide(self, trajectory_norm, normalizer, fault_spec, alpha_prod_t):
    call_count["total"] += 1
    if float(alpha_prod_t) < self.alpha_threshold:
        call_count["skipped_low_alpha"] += 1
        return trajectory_norm
    result = orig_guide(self, trajectory_norm, normalizer, fault_spec, alpha_prod_t)
    return result
eg.EmbodimentGuidance.guide = counted_guide

orig_print = print
warn_count = [0]
import builtins
def counting_print(*args, **kwargs):
    if args and isinstance(args[0], str) and "[EmbodimentGuidance] WARNING" in args[0]:
        warn_count[0] += 1
    orig_print(*args, **kwargs)
builtins.print = counting_print

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
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_warnrate")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_warnrate_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type="locked", fault_severity=None,
    n_train=0, n_test=2, test_start_seed=10000, n_envs=2,
    max_steps=80,  # short episode -- just want a warning-rate estimate, not full success rate
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

builtins.print = orig_print
print("=" * 60)
print("guide() calls total:", call_count["total"])
print("skipped due to alpha_threshold:", call_count["skipped_low_alpha"])
active_calls = call_count["total"] - call_count["skipped_low_alpha"]
print("active (non-skipped) calls:", active_calls)
print("WARNING fires (non-finite grad detected):", warn_count[0])
if active_calls > 0:
    print(f"warning rate among ACTIVE calls: {warn_count[0]/active_calls*100:.1f}%")
print("score:", log.get("test/mean_score"))
