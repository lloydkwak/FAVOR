"""
Regression test: eci_conditional_sample with fault_spec=None must produce
numerically identical output to the original DiffusionUnetHybridImagePolicy
.conditional_sample (same seed), since project_fault is identity when
fault_spec is None. This validates the manual DDPMScheduler re-implementation
(Tweedie x0 + renoise) is mathematically equivalent to scheduler.step()
for the unconstrained case, BEFORE trusting it for the constrained case.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from joint_eci_projector import eci_conditional_sample

CKPT = "/workspace/data/outputs/2026.08.10/05.53.47_train_diffusion_unet_hybrid_square_image_joint/checkpoints/latest.ckpt"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_eci_identity")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
policy = workspace.ema_model if cfg.training.use_ema else workspace.model
policy.to(torch.device("cuda:0"))
policy.eval()

B, Tp, Da = 2, policy.horizon, policy.action_dim
device = policy.device
condition_data = torch.zeros(B, Tp, Da, device=device)
condition_mask = torch.zeros(B, Tp, Da, dtype=torch.bool, device=device)

with torch.no_grad():
    gen1 = torch.Generator(device=device).manual_seed(42)
    traj_original = policy.conditional_sample(condition_data, condition_mask, generator=gen1)

    gen2 = torch.Generator(device=device).manual_seed(42)
    traj_eci = eci_conditional_sample(policy, condition_data, condition_mask, fault_spec=None, generator=gen2)

diff = (traj_original - traj_eci).abs()
print("max abs diff:", diff.max().item())
print("mean abs diff:", diff.mean().item())
print("PASS" if diff.max().item() < 1e-3 else "FAIL")
