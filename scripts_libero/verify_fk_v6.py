"""
D1 v6: panda_arm_hand.urdf (robosuite's bullet_data copy, used for pybullet
IK approximation) turned out to have link1 at z=0 for q=0, not the correct
Franka spec value of z=0.333 -- confirmed by comparing against the live
MuJoCo sim's robot0_link1 body position, which is 0.333 as expected. That
means this particular URDF file's joint origins don't match the real robot
and is not usable for FK here, regardless of how forward_kinematics is
invoked.

Trying curobo's franka_panda.urdf instead, since curobo needs accurate
kinematics for actual motion planning (unlike a pybullet IK approximation).
"""
import torch
import pytorch_kinematics as pk

URDF = "/workspace/RoboTwin/envs/curobo/src/curobo/content/assets/robot/franka_description/franka_panda.urdf"

chain = pk.build_serial_chain_from_urdf(open(URDF, "rb").read(), end_link_name="panda_hand")
print("joint names:", chain.get_joint_parameter_names())

for link_name in ["panda_link1", "panda_link3", "panda_link5", "panda_link7", "panda_hand"]:
    sub_chain = pk.build_serial_chain_from_urdf(open(URDF, "rb").read(), end_link_name=link_name)
    n = len(sub_chain.get_joint_parameter_names())
    ret = sub_chain.forward_kinematics(torch.zeros(1, n))
    pos = ret.get_matrix()[0, :3, 3].numpy()
    print("  at q=0, %-14s (n_joints=%d): pos=%s" % (link_name, n, pos))
