"""
Calibrates lambda_scale to a sensible magnitude BEFORE connecting to any real
rollout. Checks the raw gradient magnitude against the typical scale of a
normalized action trajectory, then sweeps lambda_scale to find a value where
the guidance nudge is a small FRACTION of the trajectory's own scale (not a
value that overwhelms it).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch
from embodiment_guidance import EmbodimentGuidance
from panda_kinematics import panda_fk

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
q_current = torch.tensor([[0.0, 0.3, 0.0, -2.0, 0.0, 2.0, 0.7]], dtype=torch.float32)

q_far = q_current.clone()
q_far[0, 3] += 1.5
pos_far, rot_far = panda_fk(q_far)
r_far = rot_far.squeeze(0)
rot6d_far = torch.cat([r_far[0, :], r_far[1, :]])

traj = torch.zeros(1, 4, 10)
for k in range(4):
    traj[0, k, 0:3] = pos_far.squeeze(0)
    traj[0, k, 3:9] = rot6d_far
    traj[0, k, 9] = -1.0
traj_scale = traj[..., 0:9].abs().mean().item()
print("typical |trajectory| per-element (pos+rot6d only):", traj_scale)

class DummyNormalizer:
    class _Inner:
        def unnormalize(self, x): return x
        def normalize(self, x): return x
    def __getitem__(self, k): return self._Inner()

fault_spec_locked = {'joint_idx': 3, 'q_lock': q_current[0, 3].item(), 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}

for lam in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    guidance = EmbodimentGuidance(delta_max=0.2, ik_iters=3, lambda_scale=lam)
    guidance.q_current = q_current.clone()
    nudged = guidance.guide(traj.clone(), DummyNormalizer(), fault_spec_locked, torch.tensor(0.9))
    diff = (nudged[..., 0:9] - traj[..., 0:9]).abs().mean().item()
    ratio = diff / traj_scale
    print(f"lambda_scale={lam:<8} mean|nudge| per-element={diff:.4f}   ratio to trajectory scale={ratio:.4f}")
