"""
JointActuationWrapper — thin gym.Wrapper that sits between the raw robomimic
env and FaultInjector, converting ABSOLUTE joint-target actions (q_target[7]
+ gripper[1] = 8-dim) into the per-step normalized DELTA format that
robosuite 1.2.0's JointPositionController actually expects (confirmed via
source inspection: JointPositionController has no absolute mode -- action is
always interpreted as delta from current joint position, scaled by
output_max). This is the exact closed-loop approach already verified in
test_joint_position_controller_fixes_instability.py (28mm -> 0.63mm smooth
convergence, no oscillation).
"""
import numpy as np
import gym

JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]

class JointActuationWrapper(gym.Wrapper):
    def __init__(self, env, output_max=0.05):
        super().__init__(env)
        self.output_max = output_max

    def _sim(self):
        from sim_utils import find_sim
        return find_sim(self.env)

    def _qpos_addrs(self, sim):
        return [sim.model.get_joint_qpos_addr(n) for n in JOINT_NAMES]

    def step(self, action):
        # action: (8,) = q_target(7) absolute joint target + gripper(1)
        sim = self._sim()
        qpos_addrs = self._qpos_addrs(sim)
        q_now = np.array([sim.data.qpos[a] for a in qpos_addrs])
        q_target = np.asarray(action[:7], dtype=np.float64)
        gripper = np.asarray(action[7:8], dtype=np.float64)

        delta = np.clip(q_target - q_now, -self.output_max, self.output_max)
        normalized = delta / self.output_max  # controller expects input in [-1, 1]
        low_level_action = np.concatenate([normalized, gripper]).astype(np.float32)
        return self.env.step(low_level_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
