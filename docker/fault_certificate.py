"""
Feasibility certificate ε(Q): how much does a fault's forced execution
operator Phi displace the end-effector trajectory a candidate action chunk
Q would have produced, versus what it actually produces once the locked
(or range/velocity limited) joint overrides part of it?

Built on fault_kinematics.PandaKinematics (D1-verified against the live
sim to ~1mm using curobo's franka_panda.urdf). This module adds no new
kinematics -- it only defines Phi (the fault's effect on a commanded
joint vector) and the SE(3) discrepancy this creates, batched over
(N samples, H waypoints).

Design note on WHY this targets the non-faulted joints, not the faulted
one: Phase 2 (RoboTwin) found posthoc-clipping the faulted joint alone is
IDENTICAL to B1 for locked faults (env.step() already forces q_lock, so
clipping the command to the same value changes nothing) -- proven as an
identity, not measured as a null result. The only lever is how the other
6 joints compensate, which is exactly what ε(Q) scores: it is the EE
discrepancy Phi(q) introduces, not a joint-space penalty.
"""
import torch

from fault_kinematics import PandaKinematics

JOINT_NAME_TO_IDX = {f"robot0_joint{i}": i - 1 for i in range(1, 8)}

PANDA_Q_LO = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_HI = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
PANDA_QVEL_MAX = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
CONTROL_DT = 1.0 / 20.0  # confirmed dataset env_meta, control_freq=20Hz


def apply_fault(q, joint_idx, fault_type, q_lock=None, q_lo=None, q_hi=None,
                 v_max=None, q_prev=None):
    """Phi: the fault's forced execution operator on a commanded joint
    vector, batched. q: (..., 7). Returns q with joint_idx overridden
    according to fault_type; all other entries pass through unchanged
    (identity elsewhere -- this is the whole point, see module docstring).

    fault_type: 'locked' | 'range_reduced' | 'velocity_limited'
      locked:           requires q_lock (scalar or (...,) broadcastable)
      range_reduced:    requires q_lo, q_hi for that joint (as above)
      velocity_limited: requires v_max (rad/waypoint) and q_prev (the
                         joint's actual position one waypoint back)
    """
    out = q.clone()
    if fault_type == "locked":
        out[..., joint_idx] = q_lock
    elif fault_type == "range_reduced":
        out[..., joint_idx] = torch.clamp(q[..., joint_idx], q_lo, q_hi)
    elif fault_type == "velocity_limited":
        delta = torch.clamp(q[..., joint_idx] - q_prev, -v_max, v_max)
        out[..., joint_idx] = q_prev + delta
    else:
        raise ValueError(f"unknown fault_type: {fault_type!r}")
    return out


class FeasibilityCertificate:
    """Computes ε(Q) for a batch of candidate action chunks under a given
    fault, using a D1-verified PandaKinematics instance.
    """

    def __init__(self, kin: PandaKinematics, beta=0.05, waypoint_weights=None):
        """
        kin: a PandaKinematics with set_base_transform() already called
             for the current episode's robot base pose.
        beta: rotation-distance weight (m per rad of rotation angle), see
              design doc section 2. Default 0.05 -- position error of 5cm
              is treated as comparable to a 1 rad rotation error.
        waypoint_weights: optional (H,) tensor of per-waypoint weights
              w_h. None -> uniform. A future refinement (not yet used)
              could upweight waypoints near a grasp/release.
        """
        self.kin = kin
        self.beta = beta
        self.waypoint_weights = waypoint_weights

    def _rotation_angle(self, R1, R2):
        """Geodesic rotation distance between (n,3,3) rotation matrices,
        via the rotation angle of R1^T @ R2 (arccos of the trace formula).
        Numerically safer than a raw Frobenius difference for the
        certificate's use as a real distance, though D1's verification
        used Frobenius for a quick sanity check -- that choice doesn't
        carry over here since ε(Q) needs to behave like an actual metric
        for the ranking/selection use in the design (argmin over samples).
        """
        R_rel = torch.einsum("nij,njk->nik", R1.transpose(-1, -2), R2)
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_angle)

    def score(self, Q, fault_type, joint_name, fault_params):
        """Q: (N, H, 7) candidate joint-waypoint chunks (already the arm
        part of the 8-dim action -- gripper channel excluded, it isn't
        affected by these faults and doesn't enter FK).

        fault_params: dict with the keys apply_fault() needs for
        fault_type, each broadcastable to (N, H) or scalar. For
        'velocity_limited', q_prev must be supplied as (N, H, 1) tensors
        that already account for the previous waypoint (or the pre-chunk
        joint state for h=0) -- computing that sequencing is the caller's
        responsibility, since it depends on execution order, not on this
        certificate.

        Returns: (N,) mean ε per sample, and (N, H) per-waypoint distances
        for diagnostic use.
        """
        N, H, _ = Q.shape
        joint_idx = JOINT_NAME_TO_IDX[joint_name]

        Q_flat = Q.reshape(N * H, 7)
        Q_exec_flat = apply_fault(
            Q_flat, joint_idx, fault_type,
            **{k: (v.reshape(N * H) if torch.is_tensor(v) and v.numel() > 1 else v)
               for k, v in fault_params.items()}
        )

        pos_intended, rot_intended = self.kin.forward(Q_flat)
        pos_exec, rot_exec = self.kin.forward(Q_exec_flat)

        pos_err = torch.linalg.norm(pos_intended - pos_exec, dim=-1)  # (N*H,)
        rot_err = self._rotation_angle(rot_intended, rot_exec)  # (N*H,)
        dist = pos_err + self.beta * rot_err  # (N*H,)
        dist = dist.reshape(N, H)

        if self.waypoint_weights is not None:
            w = self.waypoint_weights.to(dist.device)
            eps = (dist * w).sum(dim=1) / w.sum()
        else:
            eps = dist.mean(dim=1)

        return eps, dist
