"""
Null-space compensation: given a candidate joint waypoint q and a locked
joint j forced to displacement delta_j = Phi(q)_j - q_j, redistribute that
forced displacement into the other 6 joints so the end-effector pose is
preserved as closely as possible, using a Mahalanobis-weighted minimum-norm
solve rather than plain Euclidean minimum-norm.

WHY MAHALANOBIS, NOT EUCLIDEAN: plain min-||Delta||_2 assumes the policy's
manifold is locally isotropic in joint space, which D2 gives no reason to
believe -- a small Euclidean step can still leave the manifold if it points
in a direction the policy never varies. Weighting by Sigma^{-1}, where
Sigma is the sample covariance from the SAME batch of candidates already
drawn for selection (no extra cost), makes the compensator prefer directions
the policy itself moves in -- a cheap local manifold approximation, reusing
data already computed rather than assuming anything about the manifold's
shape.

WHY POSITION-ONLY BY DEFAULT: with 1 joint locked, the remaining 6 free
joints satisfy a FULL 6D task Jacobian (position+orientation) exactly --
6 equations, 6 unknowns, generically no freedom left for the weighting to
matter. Restricting the preserved task to position only (3 equations) keeps
3 DOF genuinely free, which is where the Mahalanobis weighting has
something to act on; task-space orientation is left to drift by whatever
the weighted solve implies, which is the right tradeoff for grasp-style
tasks where position accuracy usually matters more than wrist orientation
right up to contact. Callers that need orientation preserved too can pass
task_dims=6, at the cost of the weighting becoming a no-op (or only acting
through damping) in most configurations.

Solved via the standard weighted damped least-squares form (equivalent to
Mahalanobis-weighted minimum norm as damping -> 0), which additionally
handles near-singular free-joint Jacobians gracefully instead of failing --
a QP formulation was considered and rejected for this reason: exact
equality-constrained QP has no fallback when J_free is ill-conditioned,
which is exactly the configuration space region (elbow near a self-motion
singularity) this compensator is most likely to be invoked in.
"""
import torch

from fault_kinematics import PandaKinematics

JOINT_NAME_TO_IDX = {f"robot0_joint{i}": i - 1 for i in range(1, 8)}

PANDA_Q_LO = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_HI = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


class NullspaceCompensator:
    def __init__(self, kin: PandaKinematics, task_dims=3, damping=1e-3):
        """
        kin: a PandaKinematics with set_base_transform() already called.
        task_dims: 3 (position only, default -- see module docstring) or
                   6 (position + orientation, leaves no free DOF once one
                   joint is locked; weighting becomes largely moot).
        damping: Tikhonov damping lambda in the weighted DLS solve. Not
                 zero -- protects against singular J_free @ Sigma @ J_free^T
                 near self-motion singularities (see module docstring).
        """
        assert task_dims in (3, 6)
        self.kin = kin
        self.task_dims = task_dims
        self.damping = damping

    def compensate(self, q, joint_name, delta_j, sigma=None):
        """q: (7,) a single candidate waypoint (already selected -- this
        runs on ONE chunk, not a batch of candidates; batching would need
        per-sample Jacobians, which is more expensive than the one-shot
        selection this follows).
        joint_name: which joint is locked.
        delta_j: scalar, the forced change Phi(q)_j - q_j for that joint.
        sigma: optional (7,7) sample covariance (e.g. from the same batch
               of candidates used for selection). None -> falls back to
               identity (plain Euclidean minimum norm).

        Returns: q_compensated (7,), the adjusted waypoint with joint j
        forced to q[j] + delta_j and the other 6 adjusted to compensate.
        """
        j = JOINT_NAME_TO_IDX[joint_name]
        q = q.reshape(1, 7)

        J = self.kin.jacobian(q)[0]  # (6, 7)
        J_task = J[: self.task_dims]  # (task_dims, 7)
        J_j = J_task[:, j : j + 1]  # (task_dims, 1)
        free_idx = [k for k in range(7) if k != j]
        J_free = J_task[:, free_idx]  # (task_dims, 6)

        b = -(J_j.squeeze(-1) * delta_j)  # (task_dims,) -- the task-space
        # displacement the locked joint's own forced motion would cause,
        # which the free joints need to cancel out.

        if sigma is None:
            Sigma_free = torch.eye(6)
        else:
            sigma = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma, dtype=torch.float32)
            Sigma_free = sigma[free_idx][:, free_idx]
            # guard against a degenerate (near-zero-variance) sample batch,
            # which would make the weighting itself singular
            Sigma_free = Sigma_free + 1e-6 * torch.eye(6)

        # weighted damped least squares:
        #   Delta_free = Sigma_free @ J_free^T @ (J_free @ Sigma_free @ J_free^T + damping*I)^-1 @ b
        M = J_free @ Sigma_free @ J_free.T  # (task_dims, task_dims)
        M_damped = M + self.damping * torch.eye(self.task_dims)
        Delta_free = Sigma_free @ J_free.T @ torch.linalg.solve(M_damped, b)  # (6,)

        Delta = torch.zeros(7)
        Delta[j] = delta_j
        Delta[free_idx] = Delta_free

        q_comp = q.squeeze(0) + Delta
        # joint-limit clamp as a post-hoc safety net (not part of the
        # optimization itself -- see module docstring on why an exact QP
        # with hard limit constraints was not used here)
        q_comp = torch.clamp(q_comp, PANDA_Q_LO, PANDA_Q_HI)
        return q_comp
