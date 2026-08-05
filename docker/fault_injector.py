"""
FaultInjector — thin gym.Wrapper placed between the raw robomimic env and
RobomimicImageWrapper. Applies locked / range_reduced / velocity_limited
faults at the MuJoCo qpos/qvel/jnt_range level.

IMPORTANT: the actual fault onset value (e.g. the exact joint angle a
"locked" joint gets frozen at) is determined by the environment's random
initial state at reset() time -- it is NOT known in advance and NOT
necessarily 0.0 or any other assumed constant. This class injects that
real value into the observation dict under 'fault_info' so it can cross
the AsyncVectorEnv process boundary and reach the policy (which runs in a
separate main process and has no other way to see env-internal state).
"""
import numpy as np
import gym

PANDA_QVEL_MAX = {
    'robot0_joint1': 2.175, 'robot0_joint2': 2.175,
    'robot0_joint3': 2.175, 'robot0_joint4': 2.175,
    'robot0_joint5': 2.61,  'robot0_joint6': 2.61,
    'robot0_joint7': 2.61,
}
JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]


class FaultInjector(gym.Wrapper):
    def __init__(self, env, joint_name: str, fault_type: str, severity: float = None):
        super().__init__(env)
        assert fault_type in (None, 'locked', 'range_reduced', 'velocity_limited')
        self.joint_name = joint_name
        self.fault_type = fault_type
        self.severity = severity
        self.joint_idx = JOINT_NAMES.index(joint_name) if joint_name in JOINT_NAMES else None
        self._q_onset = None
        self._orig_jnt_range = None

    def _sim(self):
        from sim_utils import find_sim
        return find_sim(self.env)

    def _qpos_addr(self, sim):
        jid = sim.model.joint_name2id(self.joint_name)
        assert jid >= 0, f"joint {self.joint_name} not found in model"
        return sim.model.jnt_qposadr[jid], sim.model.jnt_dofadr[jid], jid

    def _fault_info_array(self):
        # [fault_active(0/1), joint_idx(0-6, -1 if none), fault_type_code, q_lock_or_center, severity]
        # fault_type_code: 0=locked, 1=range_reduced, 2=velocity_limited, -1=none
        type_code = {'locked': 0.0, 'range_reduced': 1.0, 'velocity_limited': 2.0}.get(self.fault_type, -1.0)
        active = 1.0 if self.fault_type is not None else 0.0
        joint_idx = float(self.joint_idx) if self.joint_idx is not None else -1.0
        q_val = float(self._q_onset) if self._q_onset is not None else 0.0
        sev = float(self.severity) if self.severity is not None else 0.0
        return np.array([active, joint_idx, type_code, q_val, sev], dtype=np.float32)

    def get_current_qpos(self):
        """Public RPC-able getter for the robot's ACTUAL current joint
        configuration (all 7 joints), regardless of fault state. Needed so
        FavorHybridImagePolicy can seed its q_ref continuity anchor with the
        real robot pose at episode start, instead of an arbitrary zero
        vector -- the latter actively hurt IK once joint-space regularization
        was introduced (confirmed empirically: waypoint-0 pos_err jumped
        from 0.005 to 0.28 when q_ref was a meaningless all-zeros vector)."""
        sim = self._sim()
        joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        qpos = [sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint_name2id(n)]] for n in joint_names]
        return qpos

    def get_fault_info(self):
        """Public RPC-able getter -- called via AsyncVectorEnv.call('get_fault_info')
        from the main process (where the policy lives) at any time, not tied to
        the obs/info channel timing (reset() has no info channel at all in this
        gym_util worker implementation)."""
        return {
            'joint_idx': self.joint_idx,
            'fault_type': self.fault_type,
            'severity': self.severity,
            'q_lock': float(self._q_onset) if self._q_onset is not None else None,
        }

    def _inject_fault_info(self, obs):
        if isinstance(obs, dict):
            obs = dict(obs)
            obs['fault_info'] = self._fault_info_array()
        return obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.fault_type is not None:
            sim = self._sim()
            qposadr, dofadr, jid = self._qpos_addr(sim)
            self._q_onset = sim.data.qpos[qposadr]
            if self.fault_type == 'range_reduced':
                lo, hi = sim.model.jnt_range[jid]
                s = hi - lo
                half = 0.5 * s * self.severity
                new_lo = np.clip(self._q_onset - half, lo, hi)
                new_hi = np.clip(self._q_onset + half, lo, hi)
                self._orig_jnt_range = (lo, hi)
                sim.model.jnt_range[jid] = [new_lo, new_hi]
                sim.forward()
        return self._inject_fault_info(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.fault_type is not None:
            sim = self._sim()
            qposadr, dofadr, jid = self._qpos_addr(sim)
            if self.fault_type == 'locked':
                sim.data.qpos[qposadr] = self._q_onset
                sim.data.qvel[dofadr] = 0.0
                sim.forward()
            elif self.fault_type == 'velocity_limited':
                qmax = PANDA_QVEL_MAX[self.joint_name] * self.severity
                sim.data.qvel[dofadr] = np.clip(sim.data.qvel[dofadr], -qmax, qmax)
                sim.forward()
        return self._inject_fault_info(obs), reward, done, info
