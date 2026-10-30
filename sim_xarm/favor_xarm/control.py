"""Joint command layer between the 10 Hz policy and the MuJoCo position servos.

servo  (xArm Mode 1, set_servo_angle_j): the 10 Hz target is linearly interpolated from the
       command at the start of the policy step and streamed at stream_hz; every tick's increment
       is clamped to max_joint_speed * dt (the controller runs "at the fastest speed", so the
       speed limit is the only shaping). A clamped stream lags and catches up on later steps.
online (xArm Mode 6, set_servo_angle with online re-planning): at every tick each joint moves
       towards the latest target with a velocity- and acceleration-limited profile starting from
       the current commanded velocity (an online trapezoid). A new target re-plans from the
       current commanded state, never from rest.

latency_ms delays the commanded stream (transport + controller delay) as a FIFO of ticks.

velocity_feedforward: the MuJoCo servo is tau = kp (u - q) - kv qdot. With u = q_cmd alone it lags a
moving command by (kv / kp) * v (0.1 s with the Menagerie gains, i.e. ~3 deg at 0.5 rad/s). With
feedforward the ctrl is the lead command u = q_cmd + (kv / kp) v_cmd, which makes the servo
tau = kp (q_cmd - q) + kv (v_cmd - qdot): PD tracking of the commanded trajectory (an industrial
servo's behaviour), still inside the actuator force limits.
"""
import collections
import math

import numpy as np


class JointCommandStreamer:
    def __init__(self, cfg, q0):
        c = cfg["control"]
        self.mode = c["mode"]
        assert self.mode in ("servo", "online"), self.mode
        self.policy_hz = c["policy_hz"]
        self.stream_hz = c["stream_hz"]
        self.ticks_per_step = int(round(self.stream_hz / self.policy_hz))
        assert abs(self.ticks_per_step * self.policy_hz - self.stream_hz) < 1e-9
        self.dt = 1.0 / self.stream_hz
        self.vmax = math.radians(cfg["robot"]["max_joint_speed_deg"])
        self.p_vmax = math.radians(c["planner_speed_deg"])
        self.p_amax = math.radians(c["planner_acc_deg"])
        lim = np.radians(np.array(cfg["robot"]["joint_limits_deg"], dtype=float))
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.delay_ticks = int(round(c.get("latency_ms", 0.0) / 1000.0 * self.stream_hz))
        self.ff = bool(c.get("velocity_feedforward", True))
        self.lead = np.array(c["kv"], float) / np.array(c["kp"], float)
        self.reset(q0)

    def reset(self, q0):
        self.q_cmd = np.array(q0, dtype=float)
        self.v_cmd = np.zeros(7)
        self.target = self.q_cmd.copy()
        self.seg_start = self.q_cmd.copy()
        self.k = 0
        self.fifo = collections.deque([self._ctrl_out()] * self.delay_ticks)
        self.n_clamped_ticks = 0
        self.n_target_out_of_limits = 0

    def set_target(self, q_target):
        q_target = np.asarray(q_target, dtype=float)
        if np.any(q_target < self.lo) or np.any(q_target > self.hi):
            self.n_target_out_of_limits += 1
        self.target = np.clip(q_target, self.lo, self.hi)
        self.seg_start = self.q_cmd.copy()
        self.k = 0

    def tick(self):
        """Advance one stream tick; returns the joint command applied to the servos now."""
        if self.mode == "servo":
            self.k = min(self.k + 1, self.ticks_per_step)
            desired = self.seg_start + (self.target - self.seg_start) * (self.k / self.ticks_per_step)
            step = desired - self.q_cmd
            lim = self.vmax * self.dt
            if np.any(np.abs(step) > lim + 1e-12):
                self.n_clamped_ticks += 1
            step = np.clip(step, -lim, lim)
            self.v_cmd = step / self.dt
            self.q_cmd = self.q_cmd + step
        else:
            err = self.target - self.q_cmd
            vmax = min(self.p_vmax, self.vmax)
            # Speed that still allows stopping at the target with p_amax (one tick of look-ahead).
            v_des = np.sign(err) * np.minimum(vmax, np.sqrt(2.0 * self.p_amax * np.abs(err)))
            dv = np.clip(v_des - self.v_cmd, -self.p_amax * self.dt, self.p_amax * self.dt)
            v = self.v_cmd + dv
            step = v * self.dt
            # Do not overshoot the target within a tick.
            over = np.sign(err - step) != np.sign(err)
            step = np.where(over, err, step)
            v = np.where(over, 0.0, v)
            self.v_cmd = v
            self.q_cmd = self.q_cmd + step
        self.q_cmd = np.clip(self.q_cmd, self.lo, self.hi)
        if self.delay_ticks == 0:
            return self._ctrl_out()
        self.fifo.append(self._ctrl_out())
        return self.fifo.popleft()

    def _ctrl_out(self):
        """Servo ctrl for the current command (lead command with velocity feedforward)."""
        if self.ff:
            return self.q_cmd + self.lead * self.v_cmd
        return self.q_cmd.copy()

    def hold_ctrl(self):
        return self.q_cmd.copy()
