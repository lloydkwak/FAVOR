"""
Verifies eci_conditional_sample actually ENFORCES the fault constraint in
its final output, for all three fault types (locked, range_reduced,
velocity_limited), on the native joint-space Square checkpoint. Small
batch (B=2) to minimize GPU footprint if run alongside training.
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
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_eci_constraint")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
policy = workspace.ema_model if cfg.training.use_ema else workspace.model
policy.to(torch.device("cuda:0"))
policy.eval()

B, Tp, Da = 2, policy.horizon, policy.action_dim
device = policy.device
condition_data = torch.zeros(B, Tp, Da, device=device)
condition_mask = torch.zeros(B, Tp, Da, dtype=torch.bool, device=device)

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
JOINT_IDX = 2  # joint3

def check(name, fault_spec, checker_fn):
    with torch.no_grad():
        gen = torch.Generator(device=device).manual_seed(123)
        traj = eci_conditional_sample(policy, condition_data, condition_mask,
                                       fault_spec=fault_spec, generator=gen)
    q = traj[..., :7].detach().cpu()
    ok = checker_fn(q)
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"  q[:, :, {JOINT_IDX}] sample:", q[0, :4, JOINT_IDX].tolist())
    return ok

# 1) locked
q_lock_val = 0.3
fault_locked = {'joint_idx': JOINT_IDX, 'fault_type': 'locked',
                'q_lock': q_lock_val, 'q_lo': q_lo, 'q_hi': q_hi}
check("locked", fault_locked,
      lambda q: torch.allclose(q[:, :, JOINT_IDX], torch.full_like(q[:, :, JOINT_IDX], q_lock_val), atol=1e-4))

# 2) range_reduced
narrow_lo, narrow_hi = q_lo.clone(), q_hi.clone()
narrow_lo[JOINT_IDX] = -0.2
narrow_hi[JOINT_IDX] = 0.2
fault_range = {'joint_idx': JOINT_IDX, 'fault_type': 'range_reduced',
               'q_lo': narrow_lo, 'q_hi': narrow_hi}
check("range_reduced", fault_range,
      lambda q: bool(((q[:, :, JOINT_IDX] >= -0.2 - 1e-4) & (q[:, :, JOINT_IDX] <= 0.2 + 1e-4)).all()))

# 3) velocity_limited
v_max = 0.05  # rad per intra-chunk step, deliberately tight to make violation obvious if unenforced
q_anchor = torch.zeros(B, 7)
fault_vel = {'joint_idx': JOINT_IDX, 'fault_type': 'velocity_limited',
             'v_max': v_max, 'q_anchor': q_anchor, 'q_lo': q_lo, 'q_hi': q_hi}
def vel_ok(q):
    qj = q[:, :, JOINT_IDX]
    full = torch.cat([q_anchor[:, JOINT_IDX:JOINT_IDX+1], qj], dim=1)
    deltas = (full[:, 1:] - full[:, :-1]).abs()
    return bool((deltas <= v_max + 1e-4).all())
check("velocity_limited", fault_vel, vel_ok)

# 4) no fault (identity check via constraint too -- should just pass trivially, sanity)
check("no_fault (sanity)", None, lambda q: True)
