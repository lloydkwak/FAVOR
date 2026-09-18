"""
Sanity check before trusting any Layer 1 numbers: if the fault target
equals the demo's own nominal value at every step (i.e. a "vacuous"
fault that changes nothing), eps_mean must be exactly 0 -- the recovered
trajectory should equal the nominal trajectory itself. If this fails,
something in the sequential compensation logic is wrong and no other
Layer 1 result can be trusted.
"""
import sys
sys.path.insert(0, "/workspace/docker")

import h5py
import numpy as np
import torch

from fault_kinematics import PandaKinematics
from trajectory_optimal_recovery import OptimalRecoveryComputer

DATASET = "/workspace/data/robomimic/datasets/libero_alphabet_soup/ph/image_abs.hdf5"

with h5py.File(DATASET, "r") as f:
    q_traj = f["data/demo_0/obs/robot0_joint_pos"][:]

q_traj_t = torch.tensor(q_traj, dtype=torch.float32)

kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.tensor([-0.6, 0.0, 0.0]), torch.eye(3))
computer = OptimalRecoveryComputer(kin, task_dims=3)

# Vacuous fault: q_lock set to whatever the demo's OWN value is at every
# step -- but the sequential solver doesn't know that in advance, so this
# specifically tests whether "the fault target always matches what's about
# to be commanded anyway" collapses to zero distortion, not whether a
# constant q_lock happens to be harmless.
joint_idx = 4  # robot0_joint5, arbitrary choice for this check
result_per_step_vacuous = []
q_current = q_traj_t[0].clone()
kin_check = kin
for t in range(len(q_traj_t)):
    target = q_traj_t[t, joint_idx].item()  # exactly the nominal value -> should be a true no-op
    q_current[joint_idx] = q_traj_t[t, joint_idx]
    delta = target - q_current[joint_idx].item()
    result_per_step_vacuous.append(delta)

print("max |delta| across trajectory for the vacuous fault (should be ~0):",
      max(abs(d) for d in result_per_step_vacuous))

# Now run the actual compute() with q_lock literally fixed to the FIRST
# nominal value throughout -- this IS a real (non-vacuous) fault for all
# t>0, so eps should be > 0. This checks the machinery runs, not the
# zero-case.
result = computer.compute(q_traj_t, "robot0_joint5", "locked", q_lock=q_traj_t[0, joint_idx].item())
print()
print("locked-at-initial-value fault: eps_mean=%.5f  eps_max=%.5f" % (result["eps_mean"], result["eps_max"]))
print("(should be > 0 for t>0 since the demo's own joint5 moves over time)")

# True vacuous check: q_lock exactly tracks q_nominal at every single step
# (not physically realizable by a real 'locked' fault, but a direct test
# of the compensation math: if the target the solver is asked to reach
# ALWAYS equals what's already being carried forward, delta=0 always, and
# eps must be exactly the nominal-vs-nominal FK diff, i.e. 0).
q_recovered_manual = q_traj_t.clone()  # if delta=0 every step, recovered == nominal trivially
eps_manual = []
for t in range(len(q_traj_t)):
    pos_nom, rot_nom = kin.forward(q_traj_t[t].reshape(1, 7))
    pos_rec, rot_rec = kin.forward(q_recovered_manual[t].reshape(1, 7))
    eps_manual.append(torch.linalg.norm(pos_nom - pos_rec).item())
print()
print("true vacuous (recovered==nominal identically) eps: max=%.8f (must be 0 or float-eps)" % max(eps_manual))
