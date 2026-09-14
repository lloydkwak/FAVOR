"""
Shared FK/Jacobian utilities for the null-space steering design.

URDF and world-transform choice are load-bearing, not incidental -- see D1
verification (scripts_libero/verify_fk_final.py):

  - robosuite's own bullet_data/panda_description/panda_arm_hand.urdf
    (used for its internal pybullet IK) has WRONG joint origins: link1 at
    q=0 comes out at z=0, when the live MuJoCo sim's robot0_link1 body is
    at z=0.333 (the correct Franka spec value). This is a pybullet-IK
    approximation artifact, not usable for real FK.
  - curobo's franka_panda.urdf gives the correct z=0.333 for link1 at q=0,
    and matches the live sim's robot0_right_hand position to ~1mm across
    20 random configurations once the robot0_base world transform is
    applied: world_pos = base_rot @ local_pos(q) + base_pos, where
    base_pos/base_rot are read directly from the sim (base_rot is the
    identity for these tasks, but is applied anyway rather than assumed,
    so this still works if a future task mounts the robot rotated).
"""
import os

import numpy as np
import torch
import pytorch_kinematics as pk

CUROBO_PANDA_URDF = (
    "/workspace/RoboTwin/envs/curobo/src/curobo/content/assets/"
    "robot/franka_description/franka_panda.urdf"
)


class PandaKinematics:
    """FK/Jacobian for the 7-DOF Panda arm, matched to the live MuJoCo sim
    to ~1mm (see module docstring). Batched over (n, 7) joint configs.
    """

    def __init__(self, device="cpu", urdf_path=CUROBO_PANDA_URDF):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"Panda URDF not found at {urdf_path}. This path is inside "
                "the RoboTwin clone (envs/curobo/...); if RoboTwin/ has been "
                "moved or pruned, update CUROBO_PANDA_URDF."
            )
        self.device = device
        self.chain = pk.build_serial_chain_from_urdf(
            open(urdf_path, "rb").read(), end_link_name="panda_hand"
        ).to(device=device)

    def set_base_transform(self, base_pos, base_rot):
        """base_pos: (3,) np/torch array. base_rot: (3,3) np/torch array.
        Read these from the live sim once per episode via
        sim.data.get_body_xpos("robot0_base") / get_body_xmat("robot0_base").
        """
        self.base_pos = torch.as_tensor(base_pos, dtype=torch.float32, device=self.device)
        self.base_rot = torch.as_tensor(base_rot, dtype=torch.float32, device=self.device)

    def forward(self, q):
        """q: (n, 7) joint angles. Returns world-frame (n, 3) EE positions
        and (n, 3, 3) EE rotation matrices, using the base transform set
        via set_base_transform().
        """
        if not hasattr(self, "base_pos"):
            raise RuntimeError("call set_base_transform() before forward()")
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        ret = self.chain.forward_kinematics(q)
        m = ret.get_matrix()  # (n, 4, 4)
        local_pos = m[:, :3, 3]
        local_rot = m[:, :3, :3]
        world_pos = torch.einsum("ij,nj->ni", self.base_rot, local_pos) + self.base_pos
        world_rot = torch.einsum("ij,njk->nik", self.base_rot, local_rot)
        return world_pos, world_rot

    def jacobian(self, q):
        """q: (n, 7). Returns (n, 6, 7) geometric Jacobian in the WORLD
        frame (base rotation applied to both the linear and angular
        blocks pytorch_kinematics returns in the local/base frame).
        """
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        J_local = self.chain.jacobian(q)  # (n, 6, 7), local frame
        J_lin = torch.einsum("ij,njk->nik", self.base_rot, J_local[:, :3, :])
        J_ang = torch.einsum("ij,njk->nik", self.base_rot, J_local[:, 3:, :])
        return torch.cat([J_lin, J_ang], dim=1)
