"""
RenderRecorder: optional gym.Wrapper for qualitative figures and videos.

It sits between FaultInjector and RobomimicImageWrapper, only when
FaultRobomimicImageRunner gets record_cfg. It changes nothing the policy sees:
rendering does not touch physics, and the faulted-link highlight is set and
restored around each render call, so the agentview observation is unaffected.
The episode outcome is still checked against the original sweep JSON.

Per episode (seed set by the runner's init_fn through set_record_seed):
  <out_dir>/<tag>/seed<S>_trace.npz     every episode: EE position, q, action, reward, camera
  <out_dir>/<tag>/seed<S>.mp4           record_seeds only: every control step (20 Hz), lossy
  <out_dir>/<tag>/seed<S>_frames.npz    record_seeds only: every png_stride steps, lossless
"""
import json
import os

import gym
import numpy as np

from sim_utils import find_sim

JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]
EE_SITES = ("gripper0_grip_site", "gripper0_ft_frame")


def camera_params(sim, camera_name, width, height):
    cid = sim.model.camera_name2id(camera_name)
    return dict(cam_pos=np.array(sim.data.cam_xpos[cid], dtype=np.float64).copy(),
                cam_mat=np.array(sim.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3).copy(),
                fovy=float(sim.model.cam_fovy[cid]), width=int(width), height=int(height))


def project(points, cam):
    """World points (N,3) -> pixel (u, v) in the flipped (top-row-first) image; MuJoCo camera looks along -z."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pc = (p - cam["cam_pos"]) @ cam["cam_mat"]
    z = np.maximum(-pc[:, 2], 1e-6)
    f = 0.5 * cam["height"] / np.tan(np.deg2rad(cam["fovy"]) / 2)
    return np.stack([cam["width"] / 2 + f * pc[:, 0] / z, cam["height"] / 2 - f * pc[:, 1] / z], 1)


class RenderRecorder(gym.Wrapper):
    def __init__(self, env, out_dir, tag, fault_joint_name, record_seeds=(), camera="agentview",
                 width=480, height=480, png_stride=5, fps=20, crf=14, highlight=True,
                 highlight_rgba=(0.85, 0.08, 0.10, 1.0)):
        super().__init__(env)
        self.dir = os.path.join(out_dir, tag)
        os.makedirs(self.dir, exist_ok=True)
        self.record_seeds = {int(s) for s in record_seeds}
        self.camera, self.W, self.H = camera, int(width), int(height)
        self.png_stride, self.fps, self.crf = int(png_stride), int(fps), int(crf)
        self.highlight, self.highlight_rgba = highlight, np.array(highlight_rgba, dtype=np.float32)
        self.fault_joint_name = fault_joint_name
        self._pending_seed = None
        self._ep = None

    # ---- called by the runner's init_fn (via AsyncVectorEnv.call_each) ----
    def set_record_seed(self, seed):
        self._pending_seed = int(seed)

    # ---- helpers ----
    def _sim(self):
        return find_sim(self.env)

    def _ee(self, sim):
        for s in EE_SITES:
            try:
                return np.array(sim.data.site_xpos[sim.model.site_name2id(s)], dtype=np.float64).copy()
            except Exception:
                pass
        return np.array(sim.data.get_body_xpos("robot0_right_hand"), dtype=np.float64).copy()

    def _q(self, sim):
        return np.array([sim.data.qpos[sim.model.get_joint_qpos_addr(n)] for n in JOINT_NAMES])

    def _link_geoms(self, sim, joint_name):
        if joint_name is None:
            return []
        bid = sim.model.body_name2id(joint_name.replace("joint", "link"))
        return [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]

    def render_image(self, highlight_joint="fault", camera=None, width=None, height=None):
        sim = self._sim()
        j = self.fault_joint_name if highlight_joint == "fault" else highlight_joint
        geoms = self._link_geoms(sim, j) if (self.highlight and j) else []
        saved = [(g, sim.model.geom_rgba[g].copy(), int(sim.model.geom_matid[g])) for g in geoms]
        try:
            for g, _, _ in saved:
                sim.model.geom_rgba[g] = self.highlight_rgba
                sim.model.geom_matid[g] = -1
            img = sim.render(width=width or self.W, height=height or self.H, camera_name=camera or self.camera)
        finally:
            for g, rgba, mat in saved:
                sim.model.geom_rgba[g] = rgba
                sim.model.geom_matid[g] = mat
        return np.ascontiguousarray(np.asarray(img)[::-1])

    def snapshot(self, highlight_joint=None, camera=None, width=None, height=None):
        """RPC: render the current state; also return joint anchors and the camera for annotation."""
        sim = self._sim()
        cam = camera or self.camera
        W, H = width or self.W, height or self.H
        img = self.render_image(highlight_joint=highlight_joint, camera=cam, width=W, height=H)
        anchors = np.array([sim.data.xanchor[sim.model.joint_name2id(n)] for n in JOINT_NAMES])
        axes = np.array([sim.data.xaxis[sim.model.joint_name2id(n)] for n in JOINT_NAMES])
        cp = camera_params(sim, cam, W, H)
        return dict(img=img, anchors=anchors, axes=axes, ee=self._ee(sim), q=self._q(sim),
                    cameras=list(sim.model.camera_names), **cp)

    # ---- episode bookkeeping ----
    def _begin(self):
        seed, self._pending_seed = self._pending_seed, None
        if seed is None:
            return
        sim = self._sim()
        ep = dict(seed=seed, ee=[self._ee(sim)], q=[self._q(sim)], act=[], rew=[],
                  render=seed in self.record_seeds, frames=[], frame_steps=[], writer=None)
        try:
            ep["cam"] = camera_params(sim, self.camera, self.W, self.H)
        except Exception as e:
            print(f"[render_recorder] camera {self.camera!r} not found ({e}); available: "
                  f"{list(sim.model.camera_names)}", flush=True)
            ep["render"], ep["cam"] = False, None
        self._ep = ep
        if ep["render"]:
            self._open_writer()
            self._grab(0)

    def _open_writer(self):
        import av
        path = os.path.join(self.dir, f"seed{self._ep['seed']}.mp4")
        c = av.open(path, mode="w")
        s = c.add_stream("h264", rate=self.fps)
        s.width, s.height, s.pix_fmt = self.W, self.H, "yuv420p"
        s.options = {"crf": str(self.crf)}
        self._ep["writer"] = (c, s)

    def _grab(self, step):
        import av
        img = self.render_image()
        c, s = self._ep["writer"]
        for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            c.mux(p)
        if step % self.png_stride == 0:
            self._ep["frames"].append(img)
            self._ep["frame_steps"].append(step)

    def _finalize(self):
        ep, self._ep = self._ep, None
        if ep is None:
            return
        if ep["writer"] is not None:
            c, s = ep["writer"]
            for p in s.encode():
                c.mux(p)
            c.close()
        rew = np.array(ep["rew"], dtype=np.float32)
        base = os.path.join(self.dir, f"seed{ep['seed']}")
        cam = ep["cam"] or {}
        np.savez(base + "_trace.npz", seed=ep["seed"], ee=np.array(ep["ee"]), q=np.array(ep["q"]),
                 act=np.array(ep["act"]).reshape(len(ep["act"]), -1), rew=rew,
                 success=float(rew.max()) if len(rew) else 0.0,
                 fault_joint=self.fault_joint_name, camera=self.camera,
                 **{k: v for k, v in cam.items()})
        if ep["frames"]:
            np.savez_compressed(base + "_frames.npz", frames=np.stack(ep["frames"]),
                                steps=np.array(ep["frame_steps"]))
        with open(base + "_done.json", "w") as f:
            json.dump(dict(seed=ep["seed"], steps=len(rew), success=float(rew.max()) if len(rew) else 0.0,
                           rendered=bool(ep["render"])), f)

    def flush(self):
        self._finalize()
        return True

    # ---- gym API ----
    def reset(self, **kwargs):
        self._finalize()
        obs = self.env.reset(**kwargs)
        self._begin()
        return obs

    def reset_to(self, state):
        self._finalize()
        out = self.env.reset_to(state)
        self._begin()
        return out

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        ep = self._ep
        if ep is not None:
            sim = self._sim()
            ep["ee"].append(self._ee(sim)); ep["q"].append(self._q(sim))
            ep["act"].append(np.asarray(action, dtype=np.float32).copy()); ep["rew"].append(float(reward))
            if ep["render"]:
                self._grab(len(ep["rew"]))
        return obs, reward, done, info
