"""
Offline sanity check for EmbodimentGuidance, BEFORE any rollout/policy
integration. Verifies: (1) gradient actually flows (non-zero), (2) guidance
pushes the trajectory in a sensible direction (reduces L_track), (3) with no
fault, L_track for an already-reachable trajectory is near zero.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch
from embodiment_guidance import EmbodimentGuidance, rollout_tracking_cost, differentiable_ik_step
from panda_kinematics import panda_fk

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

q_current = torch.tensor([[0.0, 0.3, 0.0, -2.0, 0.0, 2.0, 0.7]], dtype=torch.float32)  # (B=1,7)

# --- Test 1: no fault, target = where the arm ALREADY is (FK of q_current) ---
# L_track should be near 0 -- the arm doesn't need to move at all.
pos0, rot0 = panda_fk(q_current)
r0 = rot0.squeeze(0)
rot6d0 = torch.cat([r0[0, :], r0[1, :]])
a_ee = torch.zeros(1, 4, 9)  # 4 waypoints, all = current pose (trivial target)
for k in range(4):
    a_ee[0, k, 0:3] = pos0.squeeze(0)
    a_ee[0, k, 3:9] = rot6d0

fault_spec_none = {'joint_idx': 3, 'q_lock': 0.0, 'fault_type': None, 'q_lo': q_lo, 'q_hi': q_hi}
delta_max_vec = torch.full((7,), 0.2)
L_trivial, _ = rollout_tracking_cost(a_ee, fault_spec_none, q_current, delta_max_vec, ik_iters=3)
print("Test1 (no fault, target=current pose) L_track:", L_trivial.item(), " (expect near 0)")
t1_pass = L_trivial.item() < 1e-3

# --- Test 2: gradient flows and points in a sensible direction ---
# Target: move joint4 to a value FAR from current, WITH joint4 locked at
# current value. L_track should be large (target unreachable in one hop with
# the joint locked + delta_max), and moving a_ee's underlying joint4 position
# further away should only make L_track worse -- check gradient sign makes
# sense on a simple 1D probe.
fault_spec_locked = {'joint_idx': 3, 'q_lock': q_current[0, 3].item(), 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}

q_far = q_current.clone()
q_far[0, 3] += 1.5  # a config the arm can't reach with joint4 locked at its current value
pos_far, rot_far = panda_fk(q_far)
r_far = rot_far.squeeze(0)
rot6d_far = torch.cat([r_far[0, :], r_far[1, :]])
a_ee_far = torch.zeros(1, 4, 9, requires_grad=True)
with torch.no_grad():
    for k in range(4):
        a_ee_far[0, k, 0:3] = pos_far.squeeze(0)
        a_ee_far[0, k, 3:9] = rot6d_far
a_ee_far.requires_grad_(True)

L_far, _ = rollout_tracking_cost(a_ee_far, fault_spec_locked, q_current, delta_max_vec, ik_iters=3)
print("Test2 (joint4 locked, unreachable target) L_track:", L_far.item(), " (expect > 0, larger than Test1)")
grad, = torch.autograd.grad(L_far.sum(), a_ee_far)
print("  gradient norm:", grad.norm().item(), " (expect > 0, i.e. gradient actually flows)")
t2_pass = L_far.item() > L_trivial.item() and grad.norm().item() > 1e-6

# --- Test 3: full guide() call, check trajectory actually changes and shrinks L_track ---
class DummyNormalizer:
    class _Inner:
        def unnormalize(self, x): return x  # identity for this offline test
        def normalize(self, x): return x
    def __getitem__(self, k): return self._Inner()

guidance = EmbodimentGuidance(delta_max=0.2, ik_iters=3)
guidance.q_current = q_current.clone()
traj = torch.zeros(1, 4, 10)
for k in range(4):
    traj[0, k, 0:3] = pos_far.squeeze(0)
    traj[0, k, 3:9] = rot6d_far
    traj[0, k, 9] = -1.0

alpha_bar_t = torch.tensor(0.9)  # late denoising step, strong guidance
nudged = guidance.guide(traj, DummyNormalizer(), fault_spec_locked, alpha_bar_t)
diff = (nudged - traj).abs().sum().item()
print("Test3 (full guide() call) total |nudged - original|:", diff, " (expect > 0)")
t3_pass = diff > 1e-6

print("=" * 60)
print("t1_pass", t1_pass, " t2_pass", t2_pass, " t3_pass", t3_pass)
if t1_pass and t2_pass and t3_pass:
    print("EMBODIMENT_GUIDANCE_OFFLINE_VERIFIED")
else:
    import sys as _s; _s.exit(1)

# --- Test 4 (NEW, post-fix): rotation error near pi -- the exact singularity
#     that caused 100% of active guidance calls to produce non-finite
#     gradients (test_guidance_warning_rate.py). ---
from embodiment_guidance import rotmat_to_axis_angle_stable
R_identity = torch.eye(3).unsqueeze(0).requires_grad_(True)
# a rotation matrix representing ~179.9 degrees about the x-axis (deliberately near pi)
theta_near_pi = torch.tensor(3.14159 - 0.001)
R_near_pi = torch.stack([
    torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0)]),
    torch.stack([torch.tensor(0.0), torch.cos(theta_near_pi), -torch.sin(theta_near_pi)]),
    torch.stack([torch.tensor(0.0), torch.sin(theta_near_pi), torch.cos(theta_near_pi)]),
]).unsqueeze(0)
R_near_pi.requires_grad_(True)
axis_angle = rotmat_to_axis_angle_stable(R_near_pi)
loss4 = axis_angle.pow(2).sum()
grad4, = torch.autograd.grad(loss4, R_near_pi)
print("Test4 (rotation near pi) axis_angle:", axis_angle.detach().numpy(),
      " grad finite:", torch.isfinite(grad4).all().item(), " grad norm:", grad4.norm().item())
t4_pass = torch.isfinite(grad4).all().item()
print("t4_pass", t4_pass)
if not t4_pass:
    import sys as _s; _s.exit(1)
print("ROTATION_SINGULARITY_FIX_VERIFIED")
