"""
First rollout connection test for EmbodimentGuidance. Step 1: NO FAULT --
guidance is instantiated but fault_spec=None means the guide() call inside
conditional_sample is skipped entirely (per the `if self.guidance is not None
and self.fault_spec is not None` gate in favor_policy.py). This should be
bit-identical in BEHAVIOR (though not necessarily bit-identical in code path
since guidance object exists) to plain B1 -- i.e. success rate should match
OSC B1 (1.00 on Lift).
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

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_guidance_nofault")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_guidance_nofault_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type=None, fault_severity=None,
    n_train=0, n_test=5, test_start_seed=10000, n_envs=5,
    max_steps=cfg.task.env_runner.max_steps,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
    actuation_mode='osc',
)
guidance = EmbodimentGuidance(delta_max=0.2, ik_iters=3)
favor = FavorHybridImagePolicy(base_policy, fault_spec=None, env_ref=runner.env, guidance=guidance)
log = runner.run(favor)
print("EmbodimentGuidance, fault_spec=None (should skip guidance), Lift, n=5 -> test/mean_score =", log.get("test/mean_score"))
print("(OSC B1 reference: 1.00)")
