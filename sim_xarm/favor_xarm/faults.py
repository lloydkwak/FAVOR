"""Joint faults for PourEnv, with the semantics of docker/fault_injector.py (LIBERO) and of the
real-robot fault layer (FAVOR_real_robot_experiment_design.md section 4.1):

  locked : the joint is held at q_lock. The command layer overwrites the faulted joint's target
           with q_lock (real-robot software lock), and after every stream tick (100 Hz) the joint's
           qpos is reset to q_lock and its velocity to zero (simulator enforcement, as in LIBERO,
           where it was done after every 20 Hz control step).
  range  : the joint's MuJoCo limit (jnt_range) is narrowed to [lo, hi]; the command layer clips the
           target into the window (real-robot software range + reduced-mode joint range), and after
           every tick a joint outside the window is put back on its edge with zero velocity (hard stop).
Onset: None = active from reset() (lock angle / window centre = the joint's angle at home, which
is fixed, so the lock angle is the same in every trial), or 'tilt20+0.5' = the joint runs healthy
until the cup tilt computed by FK first exceeds 20 deg, then locks 0.5 s later at the angle it has
at that moment (the encoder value). get_fault_info() is what a correction method may read.
"""
import math

import mujoco
import numpy as np


class FaultLayer:
    def __init__(self, joint_idx, kind="locked", half_width=None, window=None, onset=None, tilt_deg=20.0, delay_s=0.5):
        assert kind in ("locked", "range"), kind
        self.j = int(joint_idx)
        self.kind = kind
        self.half_width = half_width
        self.window = window
        assert onset in (None, "tilt20+0.5", "tilt"), onset
        self.onset = onset
        self.tilt_th = math.radians(tilt_deg)
        self.delay = delay_s
        self._orig_range = None

    # ---------------------------------------------------------------- env hooks
    def on_reset(self, env):
        m = env.m
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{self.j + 1}")
        self.qadr, self.dadr, self.jid = env.h.arm_qpos[self.j], env.h.arm_dof[self.j], jid
        if self._orig_range is None:
            self._orig_range = m.jnt_range[jid].copy()
        m.jnt_range[jid] = self._orig_range
        self.t = 0.0
        self.t_cross = None
        self.onset_time = None
        self.active = self.onset is None
        self.q_lock = None
        self.lo = self.hi = None
        if self.active:
            self._activate(env)

    def _activate(self, env):
        q = float(env.d.qpos[self.qadr])
        self.q_lock = q
        if self.kind == "range":
            if self.window is not None:
                lo, hi = self.window
            else:
                lo, hi = q - self.half_width, q + self.half_width
            olo, ohi = self._orig_range
            self.lo, self.hi = max(lo, olo), min(hi, ohi)
            env.m.jnt_range[self.jid] = [self.lo, self.hi]
        else:
            self.lo = self.hi = q
        self.active = True
        self.onset_time = env.t

    def filter_command(self, env, q_target):
        if not self.active:
            return q_target
        q = np.array(q_target, dtype=float)
        q[self.j] = np.clip(q[self.j], self.lo, self.hi)
        return q

    def after_tick(self, env):
        dt = 1.0 / env.cfg["control"]["stream_hz"]
        self.t += dt
        if not self.active:
            if self.t_cross is None and env.cup_tilt() > self.tilt_th:
                self.t_cross = self.t
            if self.t_cross is not None and self.t >= self.t_cross + self.delay - 1e-9:
                self._activate(env)
                # Hold the lock from now on in the command stream too.
                env.ctl.q_cmd[self.j] = self.q_lock
                env.ctl.target[self.j] = self.q_lock
            return
        if self.kind == "locked":
            env.d.qpos[self.qadr] = self.q_lock
            env.d.qvel[self.dadr] = 0.0
            env.ctl.q_cmd[self.j] = self.q_lock
            mujoco.mj_forward(env.m, env.d)
        else:
            # The narrowed MuJoCo limit is soft: a servo driving into it penetrates by ~0.1 deg.
            # Make it a hard stop (as the real controller's reduced-mode range would stop the joint).
            env.ctl.q_cmd[self.j] = min(max(env.ctl.q_cmd[self.j], self.lo), self.hi)
            q = env.d.qpos[self.qadr]
            if q < self.lo or q > self.hi:
                env.d.qpos[self.qadr] = min(max(q, self.lo), self.hi)
                env.d.qvel[self.dadr] = 0.0
                mujoco.mj_forward(env.m, env.d)

    # ---------------------------------------------------------------- method interface
    def get_fault_info(self):
        if not self.active:
            return {"joint_idx": self.j, "fault_type": None, "q_lock": None, "lo": None, "hi": None}
        return {"joint_idx": self.j, "fault_type": self.kind, "q_lock": self.q_lock, "lo": self.lo, "hi": self.hi,
                "onset_time": self.onset_time}

    def restore(self, env):
        if self._orig_range is not None:
            env.m.jnt_range[self.jid] = self._orig_range
