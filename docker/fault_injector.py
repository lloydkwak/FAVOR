"""
FaultInjector — a thin gym.Wrapper placed between the raw robomimic env and
RobomimicImageWrapper. Applies locked / range_reduced / velocity_limited
faults at the MuJoCo qpos/qvel/jnt_range level, per FAVOR_Phase3 design.

Does NOT modify robosuite/robomimic/mujoco_py source. Only reads/writes
sim.data / sim.model after the official env.step() has already run.
"""
import numpy as np
import gym

# Franka Emika Panda joint velocity limits (rad/s), per official spec:
# joints 1-4: 2.175 rad/s, joints 5-7: 2.61 rad/s.
PANDA_QVEL_MAX = {
    'robot0_joint1': 2.175, 'robot0_joint2': 2.175,
    'robot0_joint3': 2.175, 'robot0_joint4': 2.175,
    'robot0_joint5': 2.61,  'robot0_joint6': 2.61,
    'robot0_joint7': 2.61,
}

class FaultInjector(gym.Wrapper):
    def __init__(self, env, joint_name: str, fault_type: str, severity: float = None):
        super().__init__(env)
        assert fault_type in (None, 'locked', 'range_reduced', 'velocity_limited')
        self.joint_name = joint_name
        self.fault_type = fault_type
        self.severity = severity
        self._q_onset = None
        self._orig_jnt_range = None

    def _sim(self):
        # robomimic EnvRobosuite -> .env is the robosuite env -> .env.sim is MjSim
        return self.env.env.sim

    def _qpos_addr(self, sim):
        jid = sim.model.joint_name2id(self.joint_name)
        assert jid >= 0, f"joint {self.joint_name} not found in model — check name against sim.model.joint_names"
        return sim.model.jnt_qposadr[jid], sim.model.jnt_dofadr[jid], jid

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
        return obs

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
            # range_reduced: jnt_range already set at reset(); MuJoCo's own
            # joint-limit constraint enforces it every physics step, no
            # per-step override needed here (matches J-PARC's approach).
        return obs, reward, done, info
