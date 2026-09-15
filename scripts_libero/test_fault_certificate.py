"""
Sanity check for fault_certificate.py, not yet a full unit test suite:
locking a joint far from a candidate's predicted value for that joint
should give a larger ε than locking it near the predicted value, for
otherwise-identical candidates. This is the property the whole selection
mechanism (argmin over ε) depends on -- if it doesn't hold, the
certificate isn't measuring what it's supposed to.
"""
import sys
sys.path.insert(0, "/workspace/docker")

import torch
from fault_kinematics import PandaKinematics
from fault_certificate import FeasibilityCertificate

kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.tensor([-0.6, 0.0, 0.0]), torch.eye(3))
cert = FeasibilityCertificate(kin, beta=0.05)

torch.manual_seed(0)
q_center = torch.tensor([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7])
H = 8
Q_base = q_center.unsqueeze(0).unsqueeze(0).repeat(1, H, 1)
Q_base += 0.05 * torch.randn(1, H, 7)  # small per-waypoint variation, like a real chunk

joint_name = "robot0_joint1"
# candidate A: joint1 predicted near a lock value close to its own predictions
q_lock_close = Q_base[0, 0, 0].item()
eps_close, _ = cert.score(Q_base, "locked", joint_name, {"q_lock": q_lock_close})

# candidate B: same Q, but the lock value is far from what was predicted
q_lock_far = q_lock_close + 1.5  # ~86 degrees away
eps_far, _ = cert.score(Q_base, "locked", joint_name, {"q_lock": q_lock_far})

print("epsilon with lock close to prediction:", eps_close.item())
print("epsilon with lock far from prediction: ", eps_far.item())
print("PASS" if eps_far.item() > eps_close.item() else "FAIL",
      "-- far-lock should score strictly worse than close-lock")

# second check: identical Q, identical fault -> identical eps (determinism)
eps_repeat, _ = cert.score(Q_base, "locked", joint_name, {"q_lock": q_lock_far})
print()
print("determinism check:", "PASS" if torch.allclose(eps_far, eps_repeat) else "FAIL")
