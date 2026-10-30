"""Domain randomisation for zero-shot sim-to-real attempts (off by default: randomization.enabled).

Sampled once per reset from the episode seed (so a demonstration is reproducible):
  lighting   : diffuse intensity and direction of the scene lights, headlight ambient
  colours    : table / bowl / cup / bead material colours (multiplicative jitter per channel)
  table      : one of several procedural table textures (built at scene compile time)
  cameras    : position (m) and orientation (deg) jitter of both cameras, fovy jitter
  beads      : sliding friction, diameter (mass per bead unchanged, total stays 50.0 g)
  servo      : kp scale (kv scaled with it, so the damping ratio and the feedforward lead stay),
               command latency (ms)
Everything is restored to nominal before each new sample.
"""
import collections
import math

import mujoco
import numpy as np

N_TABLE_TEXTURES = 6


def add_table_textures(spec, seed=1234):
    """Extra table materials the randomiser can switch between (names table_dr_k)."""
    rng = np.random.default_rng(seed)
    builtins = [mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mujoco.mjtBuiltin.mjBUILTIN_FLAT, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT]
    for k in range(N_TABLE_TEXTURES):
        base = rng.uniform(0.25, 0.85, 3)
        spec.add_texture(name=f"table_dr_tex{k}", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=builtins[k % 3],
                         rgb1=base.tolist(), rgb2=np.clip(base * rng.uniform(0.6, 0.95), 0, 1).tolist(),
                         mark=mujoco.mjtMark.mjMARK_RANDOM if k % 3 == 1 else mujoco.mjtMark.mjMARK_NONE,
                         random=0.05, markrgb=np.clip(base * 0.7, 0, 1).tolist(), width=256, height=256)
        mat = spec.add_material(name=f"table_dr{k}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"table_dr_tex{k}"
        mat.texrepeat = [float(rng.integers(2, 8))] * 2


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _small_rot_quat(rng, deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = math.radians(rng.uniform(-deg, deg))
    return np.array([math.cos(ang / 2), *(math.sin(ang / 2) * axis)])


class DomainRandomizer:
    def __init__(self, env):
        self.env = env
        m = env.m
        self.cfg = env.cfg["randomization"]
        mid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n)
        self.mats = {k: mid(n) for k, n in (("table", "table_mat"), ("bowl", "bowl_mat"), ("cup", "cup_mat"),
                                           ("particle", "particle_mat"))}
        self.table_dr = [mid(f"table_dr{k}") for k in range(N_TABLE_TEXTURES)]
        self.table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.nom = dict(mat_rgba=m.mat_rgba.copy(), light_diffuse=m.light_diffuse.copy(), light_dir=m.light_dir.copy(),
                        ambient=m.vis.headlight.ambient.copy(), cam_pos=m.cam_pos.copy(), cam_quat=m.cam_quat.copy(),
                        cam_fovy=m.cam_fovy.copy(), friction=m.geom_friction.copy(), size=m.geom_size.copy(),
                        rbound=m.geom_rbound.copy(), gain=m.actuator_gainprm.copy(), bias=m.actuator_biasprm.copy(),
                        table_mat=int(m.geom_matid[self.table_geom]),
                        diameter=env.cfg["particles"]["diameter_mm"], latency=env.cfg["control"].get("latency_ms", 0.0))
        self.last = {}

    def restore(self):
        m, n = self.env.m, self.nom
        m.mat_rgba[:] = n["mat_rgba"]; m.light_diffuse[:] = n["light_diffuse"]; m.light_dir[:] = n["light_dir"]
        m.vis.headlight.ambient[:] = n["ambient"]; m.cam_pos[:] = n["cam_pos"]; m.cam_quat[:] = n["cam_quat"]
        m.cam_fovy[:] = n["cam_fovy"]; m.geom_friction[:] = n["friction"]; m.geom_size[:] = n["size"]
        m.geom_rbound[:] = n["rbound"]; m.actuator_gainprm[:] = n["gain"]; m.actuator_biasprm[:] = n["bias"]
        m.geom_matid[self.table_geom] = n["table_mat"]
        self._set_latency(n["latency"])

    def _set_latency(self, ms):
        ctl = self.env.ctl
        ctl.delay_ticks = int(round(ms / 1000.0 * ctl.stream_hz))

    def sample(self, rng):
        self.restore()
        m, c, h = self.env.m, self.cfg, self.env.h
        L = c["lighting"]
        for i in range(m.nlight):
            m.light_diffuse[i] = self.nom["light_diffuse"][i] * rng.uniform(*L["intensity"])
            d = self.nom["light_dir"][i] + np.tan(math.radians(L["dir_jitter_deg"])) * rng.uniform(-1, 1, 3)
            m.light_dir[i] = d / np.linalg.norm(d)
        m.vis.headlight.ambient[:] = rng.uniform(*L["ambient"])
        for k, jit in (("table", c["table_rgb_jitter"]), ("bowl", c["bowl_rgb_jitter"]), ("cup", c["cup_rgb_jitter"]),
                       ("particle", c["particle_rgb_jitter"])):
            i = self.mats[k]
            m.mat_rgba[i, :3] = np.clip(self.nom["mat_rgba"][i, :3] * rng.uniform(1 - jit, 1 + jit, 3), 0, 1)
        tex = None
        if c.get("table_texture", True):
            k = int(rng.integers(-1, N_TABLE_TEXTURES))    # -1 = nominal material
            if k >= 0:
                m.geom_matid[self.table_geom] = self.table_dr[k]
                tex = k
        for cid in range(m.ncam):
            wrist = cid == h.cameras["eye_in_hand"]       # rigidly mounted: smaller jitter
            pj = c.get("wrist_camera_pos_jitter", 0.005) if wrist else c["camera_pos_jitter"]
            rj = c.get("wrist_camera_rot_jitter_deg", 1.0) if wrist else c["camera_rot_jitter_deg"]
            m.cam_pos[cid] = self.nom["cam_pos"][cid] + rng.uniform(-1, 1, 3) * pj
            m.cam_quat[cid] = _quat_mul(self.nom["cam_quat"][cid], _small_rot_quat(rng, rj))
            m.cam_fovy[cid] = self.nom["cam_fovy"][cid] + rng.uniform(-1, 1) * c["camera_fovy_jitter_deg"]
        g = h.particle_geom
        if len(g):
            m.geom_friction[g, 0] = rng.uniform(*c["particle_friction"])
            dmm = rng.uniform(*c["particle_diameter_mm"])
            m.geom_size[g, 0] = dmm / 2000.0
            m.geom_rbound[g] = dmm / 2000.0
        else:
            dmm = self.nom["diameter"]
        s = rng.uniform(*c["kp_scale"])
        for a in h.arm_act:
            m.actuator_gainprm[a, 0] = self.nom["gain"][a, 0] * s
            m.actuator_biasprm[a, 1:3] = self.nom["bias"][a, 1:3] * s
        lat = rng.uniform(*c["latency_ms"])
        self._set_latency(lat)
        self.last = dict(table_texture=tex, bead_diameter_mm=dmm, bead_friction=float(m.geom_friction[g[0], 0]) if len(g) else None,
                         kp_scale=s, latency_ms=lat)
        return self.last
