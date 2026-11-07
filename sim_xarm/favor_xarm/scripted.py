"""Scripted pouring demonstrations.

Task-space plan (cup frame: z = cup axis, x = towards the pouring lip):
  home -> above (lip over the bowl, raised) -> down to pouring height -> tilt about the lip's
  horizontal axis to theta_1 (flow starts) -> slow tilt to theta_max -> hold -> back upright ->
  retreat towards home.
Every segment uses a minimum-jerk time law. Poses are sampled at the policy rate (10 Hz) and
solved sequentially with healthy-robot pose IK (warm start, null-space pull to home), giving the
10 Hz joint targets that are recorded as actions (commanded, not measured).
Small randomisation (seeded): pouring height, lip offset, yaw of the pouring direction, tilt
angles, segment durations, hold time.

Clearance: while the cup is upright its bottom hangs (inner depth + bottom) below the lip and,
with the lip over the bowl, the cup body overlaps the bowl rim footprint. The upright cup is
therefore brought only to lip = rim + (depth + bottom + upright_clearance); the lip descends to
the pouring height during the first tilt (the cup body swings up as it turns) and rises again
while the cup comes back upright. (A first version lowered the upright cup into the rim: the
cup-bowl contact shook the beads out.)
"""
import math

import mujoco
import numpy as np

from .mjik import SiteIK


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_axis(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


DEFAULT_PARAMS = dict(
    # Defaults below; config task.scripted overrides any of them (lists = uniform ranges).
    lip_height=(0.030, 0.050),        # lip above the bowl rim while pouring (m)
    lip_offset=(-0.005, 0.020),       # lip position before the bowl centre, along the pour direction (m)
    lateral_offset=(-0.008, 0.008),   # lip offset perpendicular to the pour direction (m)
    yaw_deg=(-10.0, 10.0),            # pouring direction about world z (0 = +x, away from the robot)
    above_height=0.06,                # pre-pour approach height above the upright-clearance height (m)
    upright_clearance=0.012,          # cup bottom above the rim while upright (m)
    theta1_deg=(55.0, 65.0),          # first tilt (flow onset)
    theta_max_deg=(110.0, 125.0),     # final tilt
    t_transfer=(2.6, 3.2),            # home -> above
    t_descend=(0.9, 1.2),
    t_tilt1=(1.2, 1.6),
    t_tilt2=(2.0, 2.8),
    t_hold=(1.5, 2.5),
    t_upright=(1.4, 1.8),
    t_retreat=(1.8, 2.4),
)


class ScriptedPourPlanner:
    def __init__(self, cfg, scene, params=None):
        self.cfg = cfg
        self.h = scene
        over = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["task"].get("scripted", {}).items()}
        unknown = set(over) - set(DEFAULT_PARAMS)
        if unknown:
            raise KeyError(f"unknown task.scripted keys: {sorted(unknown)}")
        self.p = dict(DEFAULT_PARAMS, **over, **(params or {}))
        lim = np.radians(np.array(cfg["robot"]["joint_limits_deg"], dtype=float))
        self.ik = SiteIK(scene.model, scene.arm_qpos, scene.arm_dof, scene.sites["cup_mouth"], lim[:, 0], lim[:, 1])
        self.home_q = np.array(cfg["robot"]["home_q"], float)
        self.vmax_frac = 0.9
        self.dt = 1.0 / cfg["control"]["policy_hz"]
        c = cfg["cup"]
        self.lip_in_cup = np.array([c["inner_radius"] + c["wall"], 0.0, 0.0])   # relative to the mouth centre
        b = cfg["bowl"]
        self.rim_z = b.get("scale_height", 0.0) + b["bottom"] + b["depth"]
        self.cup_hang = c["inner_depth"] + c["bottom"]          # cup bottom below the lip when upright

    def _u(self, rng, key):
        v = self.p[key]
        return float(rng.uniform(*v)) if isinstance(v, (tuple, list)) else float(v)

    def sample_params(self, rng):
        return {k: self._u(rng, k) for k in self.p}

    def plan(self, bowl_xy, rng, params=None):
        P = params or self.sample_params(rng)
        p_home, R_home = self.ik.fk(self.home_q)
        yaw = math.radians(P["yaw_deg"])
        R_up = rot_z(yaw)                                      # upright cup, pour direction = R_up x
        d_pour = R_up[:, 0]
        d_lat = R_up[:, 1]
        lip_xy = np.asarray(bowl_xy) - P["lip_offset"] * d_pour[:2] + P["lateral_offset"] * d_lat[:2]
        lip_pour = np.array([lip_xy[0], lip_xy[1], self.rim_z + P["lip_height"]])
        lip_up = np.array([lip_xy[0], lip_xy[1], self.rim_z + self.cup_hang + P["upright_clearance"]])
        lip_above = lip_up + np.array([0, 0, P["above_height"]])
        axis = d_lat                                           # tilt about the horizontal axis through the lip

        def mouth_from_lip(lip, R):
            return lip - R @ self.lip_in_cup

        def tilt(theta):
            return rot_axis(axis, theta) @ R_up

        # Key poses (mouth position, cup orientation).
        lip_home = p_home + R_home @ self.lip_in_cup
        keys = {
            "home": (p_home, R_home),
            "above": (mouth_from_lip(lip_above, R_up), R_up),
            "pour_start": (mouth_from_lip(lip_up, R_up), R_up),
            "tilt1": (mouth_from_lip(lip_pour, tilt(math.radians(P["theta1_deg"]))), tilt(math.radians(P["theta1_deg"]))),
            "tilt_max": (mouth_from_lip(lip_pour, tilt(math.radians(P["theta_max_deg"]))),
                         tilt(math.radians(P["theta_max_deg"]))),
        }
        retreat_lip = 0.5 * (lip_above + lip_home)

        segs = []   # (duration, f(s) -> (mouth_pos, R))

        def lin_seg(lip_a, Ra, lip_b, Rb, T):
            qa, qb = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qa, Ra.reshape(9)); mujoco.mju_mat2Quat(qb, Rb.reshape(9))

            def f(s):
                w = min_jerk(s)
                lip = (1 - w) * lip_a + w * lip_b
                R = _slerp_mat(qa, qb, w)
                return mouth_from_lip(lip, R), R
            segs.append((T, f))

        def tilt_seg(th_a, th_b, T, lip_a=None, lip_b=None):
            lip_a = lip_pour if lip_a is None else lip_a
            lip_b = lip_pour if lip_b is None else lip_b

            def f(s):
                w = min_jerk(s)
                R = tilt(th_a + (th_b - th_a) * w)
                return mouth_from_lip((1 - w) * lip_a + w * lip_b, R), R
            segs.append((T, f))

        th1, thm = math.radians(P["theta1_deg"]), math.radians(P["theta_max_deg"])
        lin_seg(lip_home, R_home, lip_above, R_up, P["t_transfer"])
        lin_seg(lip_above, R_up, lip_up, R_up, P["t_descend"])
        tilt_seg(0.0, th1, P["t_tilt1"], lip_up, lip_pour)
        tilt_seg(th1, thm, P["t_tilt2"])
        segs.append((P["t_hold"], lambda s: (mouth_from_lip(lip_pour, tilt(thm)), tilt(thm))))
        tilt_seg(thm, 0.0, P["t_upright"], lip_pour, lip_up)
        lin_seg(lip_up, R_up, retreat_lip + np.array([0, 0, 0.03]), R_home, P["t_retreat"])

        # Sample at the policy rate and solve IK sequentially. A segment whose joint speed would
        # exceed vmax_frac of the joint speed limit is slowed down and the plan is re-solved.
        vlim = self.vmax_frac * math.radians(self.cfg["robot"]["max_joint_speed_deg"]) * self.dt
        scale = np.ones(len(segs))
        for _ in range(6):
            poses, phase = [], []
            for k, (T, f) in enumerate(segs):
                n = max(1, int(round(T * scale[k] / self.dt)))
                for i in range(1, n + 1):
                    poses.append(f(i / n))
                    phase.append(k)
            phase = np.array(phase)
            q = self.home_q.copy()
            Q, errs = [], []
            for pm, R in poses:
                q, pe, re = self.ik.solve(pm, R, q, q_posture=self.home_q, n_iter=60, posture_gain=0.02)
                Q.append(q.copy()); errs.append((pe, re))
            Q = np.array(Q)
            errs = np.array(errs)
            dq = np.abs(np.diff(np.vstack([self.home_q, Q]), axis=0)).max(axis=1)
            worst = np.array([dq[phase == k].max() for k in range(len(segs))])
            if worst.max() <= vlim:
                break
            scale = np.where(worst > vlim, scale * worst / vlim * 1.05, scale)
        key_q = {}
        for name, (pm, R) in keys.items():
            key_q[name] = self.ik.solve(pm, R, self.home_q if name == "home" else Q[-1], q_posture=self.home_q,
                                        n_iter=200)[0]
        # Sequential solutions are the reference joint waypoints for the policy-free diagnostics.
        info = dict(params=P, phase=phase, segment_time_scale=scale, ik_pos_err_max=float(errs[:, 0].max()),
                    ik_rot_err_max=float(errs[:, 1].max()), dq_max=float(np.abs(np.diff(np.vstack([self.home_q, Q]), axis=0)).max()),
                    lip_pour=lip_pour, yaw=yaw, key_poses=keys)
        return Q, info


def _slerp_mat(qa, qb, w):
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = qa + w * (qb - qa)
    else:
        th = math.acos(dot)
        q = (math.sin((1 - w) * th) * qa + math.sin(w * th) * qb) / math.sin(th)
    q /= np.linalg.norm(q)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)
