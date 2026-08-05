import torch
import math

DH_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
DH_D = [0.333, 0, 0.316, 0, 0.384, 0, 0]
DH_ALPHA = [0, -math.pi/2, math.pi/2, math.pi/2, -math.pi/2, math.pi/2, math.pi/2]
FLANGE_D = 0.107  # flange offset (verified vs robot0_right_hand, diff=0.5mm)
GRIP_SITE_D = 0.097  # additional pure-z offset from flange to gripper0_grip_site (verified via offset_local)


def dh_transform(a, d, alpha, theta):
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    row0 = torch.stack([ct, -st, zeros, a*ones], dim=-1)
    row1 = torch.stack([st*ca, ct*ca, -sa*ones, -sa*d*ones], dim=-1)
    row2 = torch.stack([st*sa, ct*sa, ca*ones, ca*d*ones], dim=-1)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def _flange_transform(q):
    batch_shape = q.shape[:-1]
    T_flange = torch.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).clone()
    T_flange[..., 2, 3] = FLANGE_D + GRIP_SITE_D  # target = grip_site, not flange
    return T_flange


# robosuite's default single-arm robot placement in the world frame.
# Verified empirically (docker/tests/test_fk_regression-era diagnostics):
# sim.data.get_body_xpos('robot0_base') = [-0.56, 0.0, 0.912], and
# sim.data.get_body_xmat('robot0_base') = identity, for the Lift task.
# NOT yet independently re-verified for Can/Square -- if IK targets for
# those tasks show a similarly-shaped systematic offset, re-check this first
# rather than assuming it carries over (per project principle: measure, don't
# assume).
WORLD_BASE_POS = torch.tensor([-0.56, 0.0, 0.912])


def compute_all_frames(q):
    """Returns T_cumulative[0..7] expressed in the WORLD frame (base offset
    baked in), since all IK targets (raw policy actions, robosuite abs-action
    convention) are in world frame. T_cumulative[0] = world<-base transform,
    T_cumulative[7] = world<-joint7 frame, before the flange/grip_site offset."""
    batch_shape = q.shape[:-1]
    base = torch.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).clone()
    base[..., :3, 3] = WORLD_BASE_POS.to(dtype=q.dtype, device=q.device)
    frames = [base]
    T = frames[0]
    for i in range(7):
        Ti = dh_transform(DH_A[i], DH_D[i], DH_ALPHA[i], q[..., i])
        T = T @ Ti
        frames.append(T)
    return frames  # length 8


def panda_fk(q):
    frames = compute_all_frames(q)
    T_ee = frames[7] @ _flange_transform(q)
    pos = T_ee[..., :3, 3]
    rot = T_ee[..., :3, :3]
    return pos, rot


def panda_jacobian(q):
    """Closed-form geometric Jacobian for a serial revolute manipulator
    parameterized with CRAIG'S MODIFIED DH convention
    (T_i = Rx(alpha_{i-1}) Tx(a_{i-1}) Rz(theta_i) Tz(d_i)).

    IMPORTANT (this is the bug that was fixed here): in modified DH, joint
    i's rotation axis is NOT frame (i-1)'s z-axis directly -- it is the
    z-axis of the intermediate frame obtained by applying ONLY the
    (alpha_{i-1}, a_{i-1}) twist/offset to frame (i-1), i.e. frame (i-1)
    with theta=0, d=0 applied. Only joint 1 has alpha_0=0, which is why the
    naive "frame(i-1)'s z-axis" version happened to work for joint 1 only
    and silently produced a wrong axis (and thus wrong Jacobian column,
    confirmed via finite-difference diagnostic) for every other joint.
    """
    frames = compute_all_frames(q)
    T_ee = frames[7] @ _flange_transform(q)
    p_ee = T_ee[..., :3, 3]

    J_v_cols = []
    J_w_cols = []
    for i in range(7):
        frame_prev = frames[i]
        theta_zero = torch.zeros_like(q[..., i])
        twist = dh_transform(DH_A[i], 0.0, DH_ALPHA[i], theta_zero)  # Rx(alpha_i) Tx(a_i) only
        frame_a = frame_prev @ twist   # correct joint-i axis frame
        z_i = frame_a[..., :3, 2]
        p_i = frame_a[..., :3, 3]
        J_v_cols.append(torch.cross(z_i, p_ee - p_i, dim=-1))
        J_w_cols.append(z_i)
    J_v = torch.stack(J_v_cols, dim=-1)
    J_w = torch.stack(J_w_cols, dim=-1)
    J = torch.cat([J_v, J_w], dim=-2)

    rot = T_ee[..., :3, :3]
    return J, p_ee, rot
