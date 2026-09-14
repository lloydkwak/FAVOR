"""
D1 v5: v4's body tree shows a clean z=0.333 jump exactly at
robot0_link0 -> robot0_link1 (matches Franka's known link1 height), so the
URDF's joint origins should already encode this. Checking with q=0 whether
pytorch_kinematics' FK reproduces that first link's z=0.333 offset at all
isolates whether the problem is in how framer FK is being invoked, or
something about this URDF copy's origin tags.
"""
import torch
import pytorch_kinematics as pk

URDF = "/opt/conda/envs/libero_dp/lib/python3.8/site-packages/robosuite/models/assets/bullet_data/panda_description/urdf/panda_arm_hand.urdf"

chain = pk.build_serial_chain_from_urdf(open(URDF, "rb").read(), end_link_name="panda_hand")
print("joint names:", chain.get_joint_parameter_names())

q0 = torch.zeros(1, 7)
ret = chain.forward_kinematics(q0)
m = ret.get_matrix()
print("FK at q=0, panda_hand position:", m[0, :3, 3].numpy())

# also check intermediate links one at a time
for link_name in ["panda_link1", "panda_link2", "panda_link3", "panda_link4",
                   "panda_link5", "panda_link6", "panda_link7", "panda_hand"]:
    sub_chain = pk.build_serial_chain_from_urdf(open(URDF, "rb").read(), end_link_name=link_name)
    n = len(sub_chain.get_joint_parameter_names())
    ret = sub_chain.forward_kinematics(torch.zeros(1, n))
    pos = ret.get_matrix()[0, :3, 3].numpy()
    print("  at q=0, %-14s (n_joints=%d): pos=%s" % (link_name, n, pos))
