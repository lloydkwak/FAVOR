"""
Layer 1: policy-independent kinematic upper bound on fault recovery.

Given a demonstration's own recorded joint trajectory q_nominal(t) and a
locked-joint fault, computes the BEST POSSIBLE reconstruction of the
non-faulted 6 joints -- via sequential null-space compensation, not any
learned policy -- that stays as close as possible (in EE SE(3) space) to
the nominal EE trajectory the demonstration actually achieved.

This answers a different question than epsilon(Q) in fault_certificate.py:
that measures how much a GIVEN candidate Q (from the policy) would be
distorted by the fault. This module instead asks "what is the best any
method (including one with perfect foresight and infinite compute) could
do here" -- a ceiling that no policy-based method (B1, select, E-C-I, or
any future method) can exceed, since it optimizes directly over joint
space with no policy distribution constraint at all.

Sequential structure: unlike nullspace_compensator.py's single-waypoint
Mahalanobis-weighted compensate(), this integrates waypoint-by-waypoint
through an entire demo, using each solved waypoint as the anchor for the
next (a null-space continuation, in the classical redundancy-resolution
sense -- see Guri & Kantor 2025, cited in project's related-work search).
Uses task_dims=3 (position-only) or task_dims=6 (position+orientation)
to directly test whether the project's position-only certificate design
(the default throughout fault_certificate.py and nullspace_compensator.py)
is itself the reason certain joints (hypothesized: robot0_joint7, closest
to the wrist) show poor recovery -- if the task_dims=6 ceiling is ALSO bad
there, the joint truly lacks recoverable freedom; if task_dims=3 ceiling
is good but task_dims=6 is bad, the certificate's position-only design is
the blind spot, not the joint's kinematics.

No diffusion policy, no GPU, no learned model anywhere in this file --
pure kinematic optimization against a demo's own recorded trajectory.
"""
import sys
sys.path.insert(0, "/workspace/docker")

import numpy as np
import torch

from fault_kinematics import PandaKinematics

JOINT_NAME_TO_IDX = {f"robot0_joint{i}": i - 1 for i in range(1, 8)}

PANDA_Q_LO = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_HI = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def geodesic_rotation_angle(R1, R2):
    """Same formula as fault_certificate.py's FeasibilityCertificate,
    duplicated here (not imported) to keep this module's only dependency
    on the project being fault_kinematics.py -- this is meant to run
    standalone, independent of anything policy-related.
    """
    R_rel = R1.transpose(-1, -2) @ R2
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle)


class OptimalRecoveryComputer:
    def __init__(self, kin: PandaKinematics, task_dims=3, damping=1e-3, beta=0.05):
        assert task_dims in (3, 6)
        self.kin = kin
        self.task_dims = task_dims
        self.damping = damping
        self.beta = beta  # rotation weight in the reported epsilon, matches
        # fault_certificate.py's default so Layer 1/2/3 numbers are directly
        # comparable on the same scale

    def _single_step_compensate(self, q, joint_idx, delta_j, sigma=None):
        """One waypoint's worth of the same Mahalanobis-weighted damped
        least-squares as nullspace_compensator.NullspaceCompensator.compensate(),
        reimplemented here (not imported) so this module has zero dependency
        on anything sample-based -- sigma defaults to identity (plain
        Euclidean min-norm), appropriate since there is no policy sample
        batch to estimate a covariance from at this layer.
        """
        q = q.reshape(1, 7)
        J = self.kin.jacobian(q)[0]  # (6, 7)
        J_task = J[: self.task_dims]
        J_j = J_task[:, joint_idx : joint_idx + 1]
        free_idx = [k for k in range(7) if k != joint_idx]
        J_free = J_task[:, free_idx]

        b = -(J_j.squeeze(-1) * delta_j)

        if sigma is None:
            Sigma_free = torch.eye(6)
        else:
            Sigma_free = sigma[free_idx][:, free_idx] + 1e-6 * torch.eye(6)

        M = J_free @ Sigma_free @ J_free.T
        M_damped = M + self.damping * torch.eye(self.task_dims)
        Delta_free = Sigma_free @ J_free.T @ torch.linalg.solve(M_damped, b)

        Delta = torch.zeros(7)
        Delta[joint_idx] = delta_j
        Delta[free_idx] = Delta_free

        q_new = q.squeeze(0) + Delta
        q_new = torch.clamp(q_new, PANDA_Q_LO, PANDA_Q_HI)
        return q_new

    def compute(self, q_nominal_traj, joint_name, fault_type, q_lock=None, q_lo=None, q_hi=None):
        """q_nominal_traj: (T, 7) the demo's own recorded joint trajectory.
        Returns: dict with 'q_recovered' (T,7), 'eps_per_step' (T,), 'eps_mean' (scalar).

        Sequential: q_recovered[0] is computed by compensating q_nominal[0]
        directly (delta_j = forced value - nominal value at that instant).
        q_recovered[t>0] compensates q_recovered[t-1] (not q_nominal[t-1])
        forward by the SAME per-step delta the fault forces at time t,
        so errors/corrections accumulate realistically across the
        trajectory rather than being independently re-solved from the
        (unreachable, since the joint is locked) nominal value every step.
        """
        joint_idx = JOINT_NAME_TO_IDX[joint_name]
        T = q_nominal_traj.shape[0]

        q_recovered = torch.zeros_like(q_nominal_traj)
        eps_per_step = torch.zeros(T)

        q_current = q_nominal_traj[0].clone()
        for t in range(T):
            q_nom_t = q_nominal_traj[t]
            if fault_type == "locked":
                target_j = q_lock
            elif fault_type == "range_reduced":
                target_j = torch.clamp(q_nom_t[joint_idx], q_lo, q_hi)
            else:
                raise ValueError(f"unsupported fault_type: {fault_type!r}")

            # delta needed to move THIS step's carried-forward joint value
            # to the fault-forced value, continuing from wherever the
            # previous step's compensation landed (not resetting to nominal)
            q_current[joint_idx] = q_nom_t[joint_idx]  # advance the faulted
            # channel's "intended" value along with the nominal trajectory
            # (it's still being commanded normally, just physically ignored)
            delta_j = (target_j - q_current[joint_idx]).item()

            q_current = self._single_step_compensate(q_current, joint_idx, delta_j)
            q_recovered[t] = q_current

            pos_nom, rot_nom = self.kin.forward(q_nom_t.reshape(1, 7))
            pos_rec, rot_rec = self.kin.forward(q_current.reshape(1, 7))
            pos_err = torch.linalg.norm(pos_nom - pos_rec, dim=-1)
            rot_err = geodesic_rotation_angle(rot_nom, rot_rec)
            eps_per_step[t] = (pos_err + self.beta * rot_err).item()

        return {
            "q_recovered": q_recovered,
            "eps_per_step": eps_per_step,
            "eps_mean": eps_per_step.mean().item(),
            "eps_max": eps_per_step.max().item(),
        }
