"""Batched FK / Jacobian of the xArm7 (pytorch_kinematics), with the same interface as
docker/fault_kinematics.py:PandaKinematics (set_base_transform, forward, jacobian, batched (n, 7)),
so the FAVOR IK modules (docker/ik_redistribution.py, ik_priority.py, ik_priority_rev.py) run
unchanged on it. It additionally exposes q_lo / q_hi (official joint limits), which those modules
use instead of the Panda limits when present.

Reference frame ("TCP"): a fixed tool transform in the link7 frame, appended to the URDF as a
fixed joint so the Jacobian is that of the tool point. from_scene() reads the transform of any
scene frame (cup_mouth, gripper_tcp, ...) from the compiled MuJoCo model, plus an optional
offset expressed in that frame (kinematics.tcp_offset), so FK matches the simulator by
construction of the tool and by the shared joint chain (verified in scripts/m3_fk_check.py).
"""
import math
import os

import numpy as np
import torch
import pytorch_kinematics as pk

from .config import resolve

DEFAULT_URDF = resolve("assets/xarm7_favor.urdf")


def _tool_joint_xml(xyz, rpy, parent="link7"):
    return (f'  <link name="tool"/>\n  <joint name="tool_fix" type="fixed">\n    <parent link="{parent}"/>\n'
            f'    <child link="tool"/>\n    <origin xyz="{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}" '
            f'rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"/>\n  </joint>\n')


def mat_to_rpy(R):
    """URDF rpy (fixed-axis X-Y-Z, R = Rz(y) Ry(p) Rx(r))."""
    R = np.asarray(R, float)
    p = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = math.atan2(-R[1, 2], R[1, 1])
        y = 0.0
    return r, p, y


class XArm7Kinematics:
    def __init__(self, device="cpu", urdf_path=DEFAULT_URDF, tool_pos=(0.0, 0.0, 0.0), tool_rot=None,
                 q_limits_deg=None):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(urdf_path)
        self.device = device
        xml = open(urdf_path).read()
        R = np.eye(3) if tool_rot is None else np.asarray(tool_rot, float)
        xml = xml.replace("</robot>", _tool_joint_xml(tool_pos, mat_to_rpy(R)) + "</robot>")
        self.chain = pk.build_serial_chain_from_urdf(xml.encode(), end_link_name="tool").to(device=device)
        self.tool_pos = np.asarray(tool_pos, float)
        self.tool_rot = R
        if q_limits_deg is None:          # limits written into the URDF (official ranges)
            lo, hi = self.chain.get_joint_limits()
        else:
            lo = [math.radians(a) for a, _ in q_limits_deg]
            hi = [math.radians(b) for _, b in q_limits_deg]
        self.q_lo = torch.tensor(lo, dtype=torch.float32)
        self.q_hi = torch.tensor(hi, dtype=torch.float32)

    # --- same interface as PandaKinematics -------------------------------------------------
    def set_base_transform(self, base_pos, base_rot):
        self.base_pos = torch.as_tensor(np.asarray(base_pos), dtype=torch.float32, device=self.device)
        self.base_rot = torch.as_tensor(np.asarray(base_rot), dtype=torch.float32, device=self.device)

    def forward(self, q):
        if not hasattr(self, "base_pos"):
            raise RuntimeError("call set_base_transform() before forward()")
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        m = self.chain.forward_kinematics(q).get_matrix()
        world_pos = torch.einsum("ij,nj->ni", self.base_rot, m[:, :3, 3]) + self.base_pos
        world_rot = torch.einsum("ij,njk->nik", self.base_rot, m[:, :3, :3])
        return world_pos, world_rot

    def jacobian(self, q):
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        J = self.chain.jacobian(q)
        J_lin = torch.einsum("ij,njk->nik", self.base_rot, J[:, :3, :])
        J_ang = torch.einsum("ij,njk->nik", self.base_rot, J[:, 3:, :])
        return torch.cat([J_lin, J_ang], dim=1)


def tool_from_scene(scene, frame, offset=(0.0, 0.0, 0.0)):
    """Pose of a scene site (cup_mouth, gripper_tcp, ...) in the link7 body frame, read from the
    compiled MuJoCo model (constant: the site is rigidly attached to link7)."""
    import mujoco
    m = scene.model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link7")
    sid = scene.sites[frame]
    Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
    Rs, ps = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
    R = Rb.T @ Rs
    p = Rb.T @ (ps - pb) + R @ np.asarray(offset, float)
    return p, R


def base_from_scene(scene):
    import mujoco
    m = scene.model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    b = scene.bodies["link_base"]
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def from_scene(scene, cfg, frame=None, device="cpu"):
    k = cfg["kinematics"]
    frame = frame or k["tcp_frame"]
    p, R = tool_from_scene(scene, frame, k.get("tcp_offset", (0, 0, 0)))
    kin = XArm7Kinematics(device=device, tool_pos=p, tool_rot=R, q_limits_deg=cfg["robot"]["joint_limits_deg"])
    kin.set_base_transform(*base_from_scene(scene))
    return kin
