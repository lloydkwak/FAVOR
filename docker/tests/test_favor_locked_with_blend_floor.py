import sys, os
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from ik_projector import ProjectWaypoints
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_blend_floor")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

q_lo_full = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi_full = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

BLEND_FLOOR = 0.15
results = {}

# --- (a) no fault, with floor -- must still keep near-1.0 success ---
projector1 = ProjectWaypoints(K=64, iters=5)
fault_spec_none = None
favor_nofault = FavorHybridImagePolicy(base_policy, fault_spec=None, projector=projector1, blend_floor=BLEND_FLOOR)
runner1 = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_bf_nofault",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint4", fault_type=None, fault_severity=None,
    n_train=0, n_test=5, n_envs=5,
    max_steps=cfg.task.env_runner.max_steps,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
)
log1 = runner1.run(favor_nofault)
results["favor_nofault_floor015"] = log1.get("test/mean_score")
print("favor_nofault (blend_floor=0.15) ->", results["favor_nofault_floor015"])

# --- (b) joint4 locked, with floor ---
fault_spec_locked = {
    'joint_idx': 3, 'q_lock': 0.0, 'fault_type': 'locked',
    'q_lo': q_lo_full, 'q_hi': q_hi_full,
}
projector2 = ProjectWaypoints(K=64, iters=5)
runner2 = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_bf_locked_j4",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint4", fault_type="locked", fault_severity=None,
    n_train=0, n_test=5, n_envs=5,
    max_steps=cfg.task.env_runner.max_steps,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
)
favor_locked = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec_locked, projector=projector2,
                                       env_ref=runner2.env, blend_floor=BLEND_FLOOR)
log2 = runner2.run(favor_locked)
results["favor_locked_j4_floor015"] = log2.get("test/mean_score")
print("favor_locked_j4 (blend_floor=0.15) ->", results["favor_locked_j4_floor015"])

print("=" * 60)
print(results)
