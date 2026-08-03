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

def panda_fk(q):
    batch_shape = q.shape[:-1]
    T = torch.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).clone()
    for i in range(7):
        Ti = dh_transform(DH_A[i], DH_D[i], DH_ALPHA[i], q[..., i])
        T = T @ Ti
    T_flange = torch.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).clone()
    T_flange[..., 2, 3] = FLANGE_D + GRIP_SITE_D  # target = grip_site, not flange
    T = T @ T_flange
    pos = T[..., :3, 3]
    rot = T[..., :3, :3]
    return pos, rot

def panda_jacobian(q):
    q = q.detach().requires_grad_(True)
    pos, rot = panda_fk(q)
    J_pos = torch.zeros(*q.shape[:-1], 3, 7, dtype=q.dtype, device=q.device)
    for i in range(3):
        grad = torch.autograd.grad(pos[..., i].sum(), q, create_graph=False, retain_graph=True)[0]
        J_pos[..., i, :] = grad
    return J_pos, pos.detach(), rot.detach()
