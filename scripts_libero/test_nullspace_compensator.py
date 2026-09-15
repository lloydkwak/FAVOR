"""
Sanity check for nullspace_compensator.py: applying the compensation should
REDUCE epsilon(Q) (fault_certificate.py's feasibility score) relative to
the uncompensated waypoint, for a locked-joint fault. This is the property
the whole point of compensation depends on -- if compensating doesn't
lower epsilon, it isn't doing anything useful, whatever its internal math
looks like.

Also checks that Mahalanobis weighting (using a directional sample
covariance) differs from plain Euclidean (identity covariance) when the
covariance is anisotropic, confirming the weighting actually has an effect
rather than silently degenerating to the same answer as identity.
"""
import sys
sys.path.insert(0, "/workspace/docker")

import torch
from fault_kinematics import PandaKinematics
from fault_certificate import FeasibilityCertificate
from nullspace_compensator import NullspaceCompensator, JOINT_NAME_TO_IDX

kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.tensor([-0.6, 0.0, 0.0]), torch.eye(3))
cert = FeasibilityCertificate(kin, beta=0.05)
comp = NullspaceCompensator(kin, task_dims=3, damping=1e-3)

torch.manual_seed(0)
q = torch.tensor([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7])
joint_name = "robot0_joint5"
j = JOINT_NAME_TO_IDX[joint_name]

# a lock value noticeably different from q[j] -- forces a real displacement
q_lock = q[j].item() - 0.4
delta_j = q_lock - q[j].item()

# uncompensated: single waypoint (H=1), identity elsewhere
Q_uncomp = q.reshape(1, 1, 7)
eps_uncomp, _ = cert.score(Q_uncomp, "locked", joint_name, {"q_lock": q_lock})
print("epsilon BEFORE compensation:", eps_uncomp.item())

q_comp = comp.compensate(q, joint_name, delta_j, sigma=None)  # identity weighting
Q_comp = q_comp.reshape(1, 1, 7)
eps_comp, _ = cert.score(Q_comp, "locked", joint_name, {"q_lock": q_lock})
print("epsilon AFTER compensation (identity weighting):", eps_comp.item())
print("PASS" if eps_comp.item() < eps_uncomp.item() else "FAIL",
      "-- compensation should reduce epsilon")

print()
print("compensated joint deltas:", (q_comp - q).round(decimals=4).tolist())
print("(joint %d should be exactly %.4f; others should be nonzero)" % (j, delta_j))

# anisotropic covariance check: does weighting change the solution?
Sigma = torch.eye(7) * 0.01
Sigma[2, 2] = 1.0  # joint3 is "cheap" to move according to this covariance
q_comp_weighted = comp.compensate(q, joint_name, delta_j, sigma=Sigma)
diff = (q_comp_weighted - q_comp).abs().sum().item()
print()
print("difference between identity- and covariance-weighted solutions:", round(diff, 5))
print("PASS" if diff > 1e-4 else "FAIL", "-- anisotropic weighting should change the solution")
