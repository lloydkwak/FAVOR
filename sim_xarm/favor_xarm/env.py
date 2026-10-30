"""PourEnv: one trial = home -> approach -> tilt/pour -> upright -> retreat, driven by 10 Hz
absolute joint targets (7-D). The fault layer (faults.py) sits in front of step().

Scale: the bowl reading is the mass of particles that have settled inside the bowl's inner
volume; cross-checked by the vertical contact force the particles exert on the bowl and the scale.
"""
import math

import mujoco
import numpy as np

from .control import JointCommandStreamer
from .scene import build_scene

G = 9.81


def mat2quat_xyzw(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return np.array([q[1], q[2], q[3], q[0]])


def quat_continuous(qs):
    """Flip signs along a sequence of quaternions so consecutive ones lie in the same hemisphere
    (q and -q are the same rotation; the canonical w >= 0 form jumps when the rotation angle passes
    180 deg, which the pouring tilt + yaw does). The first quaternion keeps its canonical sign."""
    qs = np.array(qs, dtype=np.float64, copy=True)
    for t in range(1, len(qs)):
        if np.dot(qs[t], qs[t - 1]) < 0:
            qs[t] = -qs[t]
    return qs


def grid_positions(cfg):
    t = cfg["task"]
    n, s = t["grid_n"], t["grid_size"]
    cx, cy = t["grid_center"]
    xs = cx + np.linspace(-s / 2, s / 2, n)
    ys = cy + np.linspace(-s / 2, s / 2, n)
    # index = row (x) * n + col (y)
    return np.array([[x, y] for x in xs for y in ys])


class PourEnv:
    def __init__(self, cfg, render=True, scene=None):
        self.cfg = cfg
        self.h = scene if scene is not None else build_scene(cfg)
        self.m = self.h.model
        self.d = mujoco.MjData(self.m)
        self.home_q = np.array(cfg["robot"]["home_q"], dtype=float)
        lim = np.radians(np.array(cfg["robot"]["joint_limits_deg"], dtype=float))
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self.n_sub = int(round(1.0 / cfg["control"]["stream_hz"] / self.m.opt.timestep))
        self.ctl = JointCommandStreamer(cfg, self.home_q)
        self.grid = grid_positions(cfg)
        self._renderers = {}
        self.render_enabled = render
        self.fault = None            # set by faults.FaultLayer
        self._post_tick_hooks = []
        g = self.h.geom_groups
        self._robot_geoms = np.concatenate([g["robot"], g["cup"]])
        self._is_robot = np.zeros(self.m.ngeom, bool); self._is_robot[g["robot"]] = True
        self._is_cuprob = np.zeros(self.m.ngeom, bool); self._is_cuprob[self._robot_geoms] = True
        self._is_table = np.zeros(self.m.ngeom, bool); self._is_table[g["table"]] = True
        self._is_particle = np.zeros(self.m.ngeom, bool); self._is_particle[g["particles"]] = True
        self._is_bowl = np.zeros(self.m.ngeom, bool); self._is_bowl[g["bowl"]] = True
        self._is_scale = np.zeros(self.m.ngeom, bool); self._is_scale[g["scale"]] = True
        self._is_bowlobj = np.zeros(self.m.ngeom, bool); self._is_bowlobj[np.concatenate([g["bowl"], g["scale"]])] = True
        self.self_collision_whitelist = set()
        self.t = 0.0
        self.step_count = 0
        self.randomizer = None
        if cfg.get("randomization", {}).get("enabled", False):
            from .randomization import DomainRandomizer
            self.randomizer = DomainRandomizer(self)
        self.dr_sample = {}

    # ------------------------------------------------------------------ reset
    def set_bowl_xy(self, xy):
        self.m.body_pos[self.h.bodies["scale"], :2] = xy

    def bowl_xy_for(self, rng, grid_index=None):
        t = self.cfg["task"]
        if grid_index is not None:
            return self.grid[grid_index].copy()
        if t["placement"] == "grid":
            return self.grid[rng.integers(len(self.grid))].copy()
        s = t["grid_size"]
        return np.array(t["grid_center"]) + rng.uniform(-s / 2, s / 2, size=2)

    def _set_arm(self, q):
        self.d.qpos[self.h.arm_qpos] = q
        self.d.qvel[self.h.arm_dof] = 0.0
        self.d.ctrl[self.h.arm_act] = q

    def _spawn_particles(self, rng):
        c, p = self.cfg["cup"], self.cfg["particles"]
        n = len(self.h.particle_qpos)
        r = getattr(self, "_bead_diameter_mm", p["diameter_mm"]) / 2000.0
        sp = 2 * r * 1.04
        R_in = c["inner_radius"] - r - 0.0005
        pts = []
        z = r + 0.0005
        layer = 0
        while len(pts) < n:
            off = (sp / 2) * (layer % 2)
            rows = np.arange(-R_in, R_in + 1e-9, sp * math.sqrt(3) / 2)
            for ri, y in enumerate(rows):
                xs = np.arange(-R_in, R_in + 1e-9, sp) + off + (sp / 2) * (ri % 2)
                for x in xs:
                    if x * x + y * y <= R_in * R_in and len(pts) < n:
                        pts.append((x, y, z))
            z += sp * 0.82
            layer += 1
        pts = np.array(pts)
        pts[:, :2] += rng.uniform(-p["spawn_jitter"], p["spawn_jitter"], size=(n, 2))
        mujoco.mj_kinematics(self.m, self.d)
        bid = self.h.bodies["cup"]
        P, Rc = self.d.xpos[bid].copy(), self.d.xmat[bid].reshape(3, 3).copy()
        w = pts @ Rc.T + P
        for i, a in enumerate(self.h.particle_qpos):
            self.d.qpos[a:a + 3] = w[i]
            self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.d.qvel[self.h.particle_dof[:, None] + np.arange(6)] = 0.0
        return float(pts[:, 2].max() + r)

    def reset(self, seed=0, grid_index=None, bowl_xy=None, settle=True):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        xy = np.asarray(bowl_xy, float) if bowl_xy is not None else self.bowl_xy_for(rng, grid_index)
        self.set_bowl_xy(xy)
        self.bowl_xy = xy
        self._set_arm(self.home_q)
        if self.h.gripper_act >= 0:
            self.d.ctrl[self.h.gripper_act] = self.cfg["robot"]["gripper"]["closed_ctrl"]
        if self.randomizer is not None:
            self.dr_sample = self.randomizer.sample(np.random.default_rng([seed, 77]))
            self._bead_diameter_mm = self.dr_sample["bead_diameter_mm"]
        self.fill_height = self._spawn_particles(rng)
        mujoco.mj_forward(self.m, self.d)
        self.ctl.reset(self.home_q)
        if settle:
            n = int(round(self.cfg["particles"]["settle_time"] / self.m.opt.timestep))
            for _ in range(n):
                self.d.ctrl[self.h.arm_act] = self.home_q
                mujoco.mj_step(self.m, self.d)
        self.t = 0.0
        self.step_count = 0
        self.safety = {"table_contact": 0, "self_collision": 0, "joint_limit": 0, "first": None}
        self.cup_bowl_contact = 0          # informational (not a safety violation in the task spec)
        self.max_tilt = 0.0
        if self.fault is not None:
            self.fault.on_reset(self)
        return self.observe()

    # ------------------------------------------------------------------ step
    def step(self, q_target):
        """One policy step (1 / policy_hz s). q_target: (7,) absolute joint target."""
        q_target = np.asarray(q_target, dtype=float)
        if self.fault is not None:
            q_target = self.fault.filter_command(self, q_target)
        self.ctl.set_target(q_target)
        for _ in range(self.ctl.ticks_per_step):
            q_cmd = self.ctl.tick()
            self.d.ctrl[self.h.arm_act] = q_cmd
            for _ in range(self.n_sub):
                mujoco.mj_step(self.m, self.d)
            if self.fault is not None:
                self.fault.after_tick(self)
            self._check_safety()
        self.t += 1.0 / self.cfg["control"]["policy_hz"]
        self.step_count += 1
        self.max_tilt = max(self.max_tilt, self.cup_tilt())
        return self.observe()

    def hold(self, seconds):
        """Physics with the current command held (weighing / settling)."""
        n = int(round(seconds * self.cfg["control"]["stream_hz"]))
        for _ in range(n):
            self.d.ctrl[self.h.arm_act] = self.ctl.hold_ctrl()
            for _ in range(self.n_sub):
                mujoco.mj_step(self.m, self.d)
            if self.fault is not None:
                self.fault.after_tick(self)

    # ------------------------------------------------------------------ safety
    def _check_safety(self):
        nc = self.d.ncon
        q = self.d.qpos[self.h.arm_qpos]
        marg = math.radians(self.cfg["task"]["safety"]["joint_limit_margin_deg"])
        if np.any(q < self.q_lo - 1e-3 + marg) or np.any(q > self.q_hi + 1e-3 - marg):
            self._flag("joint_limit")
        if nc == 0:
            return
        gg = self.d.contact.geom[:nc]
        g1, g2 = gg[:, 0], gg[:, 1]
        a_rob, b_rob = self._is_cuprob[g1], self._is_cuprob[g2]
        tab = (a_rob & self._is_table[g2]) | (b_rob & self._is_table[g1])
        if tab.any():
            self._flag("table_contact")
        if ((a_rob & self._is_bowlobj[g2]) | (b_rob & self._is_bowlobj[g1])).any():
            self.cup_bowl_contact += 1
        both = self._is_robot[g1] & self._is_robot[g2]
        cup_rob = (self._is_robot[g1] & np.isin(g2, self.h.geom_groups["cup"])) | \
                  (self._is_robot[g2] & np.isin(g1, self.h.geom_groups["cup"]))
        sc = both | cup_rob
        if sc.any():
            for a, b in zip(g1[sc], g2[sc]):
                key = (min(a, b), max(a, b))
                if key not in self.self_collision_whitelist:
                    self._flag("self_collision")
                    break

    def _flag(self, k):
        self.safety[k] += 1
        if self.safety["first"] is None:
            self.safety["first"] = (k, round(self.t, 2))

    def safety_violated(self):
        return any(self.safety[k] > 0 for k in ("table_contact", "self_collision", "joint_limit"))

    # ------------------------------------------------------------------ metrics
    def particle_pos(self):
        return self.d.qpos[self.h.particle_qpos[:, None] + np.arange(3)]

    def particle_speed(self):
        return np.linalg.norm(self.d.qvel[self.h.particle_dof[:, None] + np.arange(3)], axis=1)

    def in_bowl_mask(self):
        b = self.cfg["bowl"]
        P = self.particle_pos()
        bid = self.h.bodies["bowl"]
        c = self.d.xpos[bid]
        rel = P - c
        r = np.linalg.norm(rel[:, :2], axis=1)
        return (r < b["inner_radius"]) & (rel[:, 2] > b["bottom"] - 1e-3) & (rel[:, 2] < b["bottom"] + b["depth"] + 0.002)

    def in_cup_mask(self):
        c = self.cfg["cup"]
        P = self.particle_pos()
        bid = self.h.bodies["cup"]
        rel = (P - self.d.xpos[bid]) @ self.d.xmat[bid].reshape(3, 3)
        r = np.linalg.norm(rel[:, :2], axis=1)
        return (r < c["inner_radius"]) & (rel[:, 2] > -1e-3) & (rel[:, 2] < c["inner_depth"] + 0.002)

    def bowl_contact_mass(self):
        """Vertical contact force exerted by particles on the bowl + scale, / g (kg)."""
        f6 = np.zeros(6)
        fz = 0.0
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            g1, g2 = con.geom
            if (self._is_particle[g1] and self._is_bowlobj[g2]) or (self._is_particle[g2] and self._is_bowlobj[g1]):
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                fw = con.frame.reshape(3, 3).T @ f6[:3]       # contact frame -> world (force on geom2)
                # Force on the bowl: if the bowl is geom2, +fw; if geom1, -fw.
                fz += fw[2] if self._is_bowlobj[g2] else -fw[2]
        return -fz / G

    def metrics(self):
        total = self.cfg["particles"]["total_mass_g"]
        mp = self.h.particle_mass * 1000.0
        settled = self.particle_speed() < self.cfg["task"]["particle_speed_settled"]
        bowl = self.in_bowl_mask()
        cup = self.in_cup_mask()
        bowl_g = float((bowl & settled).sum() * mp)
        cup_g = float(cup.sum() * mp)
        f = bowl_g / total
        return {"f": f, "spill_g": total - bowl_g - cup_g, "bowl_g": bowl_g, "cup_g": cup_g,
                "bowl_unsettled": int((bowl & ~settled).sum()),
                "bowl_force_g": 1000.0 * self.bowl_contact_mass(),
                "safety_violation": bool(self.safety_violated()),
                "success": bool(f >= self.cfg["task"]["success_fraction"] and not self.safety_violated()),
                "max_tilt_deg": math.degrees(self.max_tilt), "robot_bowl_contact_ticks": self.cup_bowl_contact, **{f"safety_{k}": v for k, v in self.safety.items()}}

    # ------------------------------------------------------------------ observation
    def site_pose(self, name):
        sid = self.h.sites[name]
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def cup_tilt(self):
        R = self.d.xmat[self.h.bodies["cup"]].reshape(3, 3)
        return math.acos(max(-1.0, min(1.0, R[2, 2])))

    def joint_pos(self):
        return self.d.qpos[self.h.arm_qpos].copy()

    def _raw_render(self, cam, H, W, fovy=None):
        key = (H, W)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.m, H, W)
        r = self._renderers[key]
        cid = self.h.cameras[cam]
        if fovy is not None:
            old = self.m.cam_fovy[cid]
            self.m.cam_fovy[cid] = fovy
        r.update_scene(self.d, camera=cid)
        img = r.render()
        if fovy is not None:
            self.m.cam_fovy[cid] = old
        return img

    def render(self, cam, hw=None):
        """RGB frame (H, W, 3) uint8, row 0 = top. hw defaults to cameras.render_size.
        Pinhole intrinsics fx/fy/cx/cy (pixels at intrinsics.width x height) are applied exactly:
        a square-pixel image with focal fy is rendered with margins and cropped/resized so that
        u = fx X/Z + cx, v = fy Y/Z + cy (OpenCV convention), scaled to hw."""
        import cv2
        H, W = hw or tuple(self.cfg["cameras"]["render_size"])
        intr = self.cfg["cameras"][cam]["intrinsics"]
        if "fx" not in intr:
            return self._raw_render(cam, H, W)
        s = H / intr["height"]
        fx, fy, cx, cy = intr["fx"] * W / intr["width"], intr["fy"] * s, intr["cx"] * W / intr["width"], intr["cy"] * s
        f = fy                                            # square pixels in the raw render
        x0 = cx * f / fx                                  # raw columns left / right of the principal point
        x1 = (W - cx) * f / fx
        half_w = int(math.ceil(max(x0, x1))) + 1
        half_h = int(math.ceil(max(cy, H - cy))) + 1
        Hp, Wp = 2 * half_h, 2 * half_w
        fovy = math.degrees(2 * math.atan(0.5 * Hp / f))
        raw = self._raw_render(cam, Hp, Wp, fovy=fovy)
        top = int(round(half_h - cy))
        crop = raw[top:top + H, int(round(half_w - x0)):int(round(half_w + x1))]
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA if fx < f else cv2.INTER_LINEAR)

    def policy_image(self, cam):
        import cv2
        img = self.render(cam)
        H, W = img.shape[:2]
        s = min(H, W)
        y0, x0 = (H - s) // 2, (W - s) // 2
        sq = img[y0:y0 + s, x0:x0 + s]
        n = self.cfg["cameras"]["policy_size"]
        return cv2.resize(sq, (n, n), interpolation=cv2.INTER_AREA)

    def observe(self, images=None):
        images = self.render_enabled if images is None else images
        p, R = self.site_pose(self.cfg["observation"]["eef_frame"])
        q = mat2quat_xyzw(R)
        if self.step_count > 0 and getattr(self, "_last_quat", None) is not None and np.dot(q, self._last_quat) < 0:
            q = -q                    # same rule as quat_continuous(); restarts at every reset()
        self._last_quat = q
        obs = {"robot0_eef_pos": p.astype(np.float32), "robot0_eef_quat": q.astype(np.float32),
               "robot0_joint_pos": self.joint_pos().astype(np.float32)}
        if images:
            obs["agentview_image"] = self.policy_image("agentview")
            obs["robot0_eye_in_hand_image"] = self.policy_image("eye_in_hand")
        return obs

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}
