"""Healthy-robot pose IK on the MuJoCo model itself (site Jacobians), used by the scripted
demonstration planner and to solve the home pose. Not one of the FAVOR fault-correction methods
(those live in docker/ik_*.py and are driven through XArm7Kinematics)."""
import mujoco
import numpy as np


def rotvec_err(R_target, R):
    """Rotation vector of R_target R^T (world frame)."""
    E = R_target @ R.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, E.reshape(9))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class SiteIK:
    def __init__(self, model, arm_qpos, arm_dof, site_id, q_lo, q_hi):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.asarray(arm_qpos)
        self.dadr = np.asarray(arm_dof)
        self.site = site_id
        self.lo, self.hi = np.asarray(q_lo), np.asarray(q_hi)
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jac(self, q):
        self.fk(q)
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
        return np.vstack([self.jp[:, self.dadr], self.jr[:, self.dadr]])

    def solve(self, p_t, R_t, q0, q_posture=None, n_iter=200, damping=1e-3, w_rot=1.0,
              posture_gain=0.05, max_step=0.1, tol_p=1e-5, tol_r=1e-4):
        """Damped least squares on the full pose with a null-space pull towards q_posture.
        Returns (q, pos_err_m, rot_err_rad)."""
        q = np.array(q0, dtype=float)
        qp = q.copy() if q_posture is None else np.asarray(q_posture, float)
        W = np.diag([1, 1, 1, w_rot, w_rot, w_rot])
        for _ in range(n_iter):
            p, R = self.fk(q)
            e = np.concatenate([p_t - p, rotvec_err(R_t, R)])
            if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_r:
                break
            J = self.jac(q)
            Jw = W @ J
            JJt = Jw @ Jw.T + damping ** 2 * np.eye(6)
            Jpinv = Jw.T @ np.linalg.solve(JJt, np.eye(6))
            dq = Jpinv @ (W @ e)
            N = np.eye(7) - Jpinv @ Jw
            dq += N @ (posture_gain * (qp - q))
            n = np.abs(dq).max()
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(rotvec_err(R_t, R)))
