"""
FaultInjector (SAPIEN / RoboTwin 2.0 port) -- wraps envs.robot.robot.Robot to
apply locked / range_reduced / velocity_limited faults at the SAPIEN
articulation level, on ONE named joint of ONE arm ("left" or "right").

Ported from the MuJoCo/robosuite FaultInjector (docker/fault_injector.py).
Design choices preserved from that version:
  - `severity` is a FRACTION (0..1) of the joint's normal range/speed, not
    an absolute physical value. This matches the loss-of-effectiveness (LOE)
    convention used in the fault-tolerant-control literature and keeps fault
    severity comparable across joints whose physical ranges differ wildly.
  - The exact fault onset value (the qpos a "locked" joint freezes at) is
    NOT known in advance -- it depends on wherever the arm happens to be
    when the fault activates. This class captures that value at activation
    time rather than assuming 0.0 or any other constant.
  - get_fault_info() / get_current_qpos() are kept as the same public,
    RPC-able getters the native joint policy (NativeJointPolicy) already
    expects from the robosuite era, so downstream code does not need to
    change its calling convention.

Key structural difference from the MuJoCo version:
  - RoboTwin does not let you write to sim.data.qpos and have that be the
    end of the story -- envs/robot/robot.py drives each arm joint with a
    SAPIEN PD drive (stiffness/damping set once in Robot.init_joints()).
    Every control step, Robot.set_arm_joints(target_position,
    target_velocity, arm_tag) pushes new drive targets. If we only wrote
    qpos/qvel directly, the PD drive would immediately start pulling the
    joint back toward whatever target_position was last commanded.
    So this wrapper hooks Robot.set_arm_joints() itself (the single choke
    point all 6 call sites in envs/_base_task.py funnel through) and
    rewrites target_position / target_velocity BEFORE they reach the drive,
    rather than fighting the drive after the fact.
  - Locking is therefore implemented as: freeze the recorded onset angle as
    both the position AND velocity target for that one joint index, every
    step, regardless of what the planner asked for.
  - range_reduced is implemented as a SAPIEN joint limit (joint.set_limits),
    which is the direct analogue of MuJoCo's sim.model.jnt_range -- the
    physics engine itself will not let the joint go outside it, independent
    of what the drive target says.
  - velocity_limited clips the requested POSITION sequence's per-step delta
    (not just target_velocity -- see _apply_step_fault for why clipping
    velocity alone does not work on this robot's strong position PD drive).

Usage:
    injector = FaultInjector(robot, arm_tag="left", joint_name="fl_joint4",
                              fault_type="range_reduced", severity=0.5)
    injector.attach()   # monkeypatches robot.set_arm_joints
    ...
    injector.get_fault_info()
    injector.get_current_qpos()
    injector.detach()   # restores original set_arm_joints, restores joint limits
"""
import numpy as np

# -----------------------------------------------------------------------
# Joint name tables for the aloha-agilex embodiment (env_cfg_type
# "aloha_agilex", assets dir "aloha-agilex"). 6 arm joints per side, per
# assets/embodiments/aloha-agilex/config.yml `arm_joints_name`.
# -----------------------------------------------------------------------
LEFT_ARM_JOINTS = ["fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6"]
RIGHT_ARM_JOINTS = ["fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6"]

# -----------------------------------------------------------------------
# Physical joint limits and max velocities.
#
# SUPERSEDED APPROACH (do not resurrect): earlier versions of this file
# sourced these from real-stanford/arx5-sdk's models/X5.urdf, on the
# assumption that RoboTwin's own urdf/arx5_description_isaac.urdf only had
# placeholder limits ([-10, 10] rad on every joint) and a vendor-adjacent
# hardware repo would have the real numbers. That assumption was WRONG:
# direct SAPIEN load-and-query of RoboTwin's own aloha-agilex URDF showed
# ALL SIX arm joints (fl_/fr_ joint1-6) load with [-10, 10] limits -- not
# just joint1. RoboTwin's simulator enforces NO physical position limit on
# any arm joint; only the gripper joints (7/8, [0, 0.04765]) have a real
# constraint. X5.urdf is a different robot variant/config and its
# per-joint limits do not describe what this simulator actually runs
# (confirmed empirically: joint1 and joint6 both show empirical training
# data excursions of 4-7 rad, physically impossible under X5.urdf's
# assumed +-1.57-3.14 rad limits).
#
# CURRENT APPROACH: use the EMPIRICAL range each joint was actually
# observed to occupy across the demo_clean training data for our three
# priority tasks (grab_roller, adjust_bottle, handover_mic; 23,008 frames
# combined), computed directly from
# XPolicyLab/policy/DP/data/demo_clean-<task>-aloha_agilex-joint.zarr's
# 'state' field. This is NOT the joint's true physical range (which the
# simulator does not enforce and no reliable spec was found for) -- it is
# "the range expert demonstrations for these three tasks actually used."
# That is a meaningfully different, narrower quantity, and the
# family/scale of tasks that get added later could observe wider ranges
# than these three did. Treat these bounds as task-set-specific, not
# robot-specific, and recompute if the task set changes materially.
#
# joint1 is EXCLUDED from range_reduced experiments regardless of its
# empirical range. Its observed excursion (-7.34 to 0.0 rad, over 2 full
# turns) reflects the complete absence of any physical constraint in the
# simulator PLUS whatever a given task's motion plan happened to need --
# not a stable "normal operating envelope" the way joints 2-6 are (their
# empirical spans are all under 2*pi, physically plausible for a real
# revolute joint even without a simulator-enforced limit). A
# range_reduced fault on joint1 would be reducing a fraction of an
# arbitrary, motion-plan-dependent quantity, not a meaningful hardware
# constraint.
# -----------------------------------------------------------------------
JOINT_LIMITS_RAD = {
    # joint_name: (lower, upper), empirical across 23,008 frames /
    # grab_roller+adjust_bottle+handover_mic demo_clean data.
    # None means "no trustworthy normal-range denominator" (joint1 only).
    "joint1": None,
    "joint2": (0.0, 2.5121),      # left/right combined envelope (left: 0..2.5121, right: 0..2.4755)
    "joint3": (0.0, 2.4937),      # (left: 0..2.4937, right: 0..2.3587)
    "joint4": (-1.6074, 1.5284),  # (left: -1.6074..1.5284, right: -1.4440..1.1621)
    "joint5": (-0.8292, 0.8490),  # (left: -0.6543..0.8490, right: -0.8292..0.7102)
    "joint6": (-5.5697, 3.4248),  # WIDE -- see caveat below
}

# joint6's combined empirical span (-5.5697 to 3.4248, ~9.0 rad total) is
# itself suspicious for the same reason joint1 is excluded: no real
# per-joint URDF constraint exists for it either (confirmed: [-10,10] in
# SAPIEN for fl_joint6/fr_joint6 too), and -5.57 rad is over 1.75 full
# turns, well beyond a plausible single-turn wrist joint range. It is
# being KEPT (not excluded) here because its magnitude, while large, is
# still bounded and reproducible across all three tasks (not a one-off
# outlier), and because dropping it would leave a 4th arm joint without a
# range_reduced fault condition -- but this is a judgment call, not a
# settled fact, and should be revisited (e.g. by checking per-task rather
# than combined ranges, or watching whether it behaves anomalously in
# later fault-sweep results) before leaning heavily on joint6 results.


JOINT_VEL_MAX_RAD_S = {
    # joint_name: max velocity (rad/s), empirical 99th-percentile |Δqpos * 15Hz|
    # across 22,858 frame-to-frame samples (episode-boundary-safe) from the
    # same 3-task demo_clean data as JOINT_LIMITS_RAD above. p99 (not raw
    # max) was used because raw max is sensitive to single motion-plan
    # discontinuities (e.g. joint6 left showed a 3.0 rad/s spike vs a 1.77
    # rad/s p99) -- p99 better represents sustained normal-operation speed,
    # which is what `severity` fractions should be scaling. Where left/
    # right values differed, the larger (more conservative) of the two was
    # kept, since this table is generic per joint-number, not per-arm.
    # Supersedes the earlier X5.urdf-derived values for the same reason
    # JOINT_LIMITS_RAD does -- see that constant's docstring: RoboTwin's
    # simulator enforces no real velocity limit on any arm joint, so no
    # URDF source (any URDF) describes what this system actually does.
    "joint1": 0.598,  # p99, from max(left 0.598, right 0.541)
    "joint2": 1.611,  # from max(left 1.611, right 1.528)
    "joint3": 1.386,  # from max(left 1.386, right 1.295)
    "joint4": 1.572,  # from max(left 1.572, right 1.421)
    "joint5": 0.618,  # from max(left 0.618, right 0.436)
    "joint6": 1.771,  # from max(left 1.771, right 1.499)
}


def _generic_joint_key(joint_name: str) -> str:
    """Map 'fl_joint4' / 'fr_joint4' -> 'joint4' for table lookups."""
    for prefix in ("fl_", "fr_"):
        if joint_name.startswith(prefix):
            return joint_name[len(prefix):]
    raise ValueError(f"Unrecognized joint name '{joint_name}' (expected fl_/fr_ prefix)")


class FaultInjector:
    def __init__(self, robot, arm_tag: str, joint_name: str, fault_type: str, severity: float = None,
                 dt: float = 0.004):
        assert arm_tag in ("left", "right")
        assert fault_type in (None, "locked", "range_reduced", "velocity_limited")
        if fault_type == "range_reduced":
            assert severity is not None and 0.0 < severity <= 1.0, \
                "range_reduced requires severity in (0, 1]"
        if fault_type == "velocity_limited":
            assert severity is not None and 0.0 < severity <= 1.0, \
                "velocity_limited requires severity in (0, 1]"

        self.robot = robot
        self.arm_tag = arm_tag
        self.joint_name = joint_name
        self.fault_type = fault_type
        self.severity = severity
        # dt: seconds per set_arm_joints() call. Measured directly from
        # SAPIEN (task.scene.get_timestep() == 0.004s / 250Hz) -- NOT
        # guessed, and NOT the same as the 15Hz save_freq used when
        # computing JOINT_VEL_MAX_RAD_S from recorded demo data (that
        # downsampling rate only affects how the empirical rad/s estimate
        # was computed, not the rate the physics engine actually steps
        # at, which is what matters for clipping a per-call position
        # delta here). Pass a different value explicitly if a task's
        # scene ever uses a different timestep.
        self.dt = dt

        joint_list = LEFT_ARM_JOINTS if arm_tag == "left" else RIGHT_ARM_JOINTS
        assert joint_name in joint_list, f"{joint_name} not an arm joint for arm_tag={arm_tag}"
        self.joint_idx = joint_list.index(joint_name)

        generic_key = _generic_joint_key(joint_name)
        if fault_type == "range_reduced" and JOINT_LIMITS_RAD.get(generic_key) is None:
            raise ValueError(
                f"range_reduced fault requested on {joint_name}, but no trustworthy "
                f"physical range is known for {generic_key} (joint1 is excluded from "
                f"range_reduced experiments -- see module docstring)."
            )
        self._normal_limits = JOINT_LIMITS_RAD.get(generic_key)
        self._normal_vel_max = JOINT_VEL_MAX_RAD_S.get(generic_key)

        self._q_onset = None          # captured at first activation (locked)
        self._vel_anchor = None       # running clipped-position anchor (velocity_limited)
        self._sapien_joint = None     # sapien.pysapien.physx.PhysxArticulationJoint
        self._original_limits = None  # to restore on detach()
        self._original_set_arm_joints = None
        self._attached = False

        # Optional lightweight verification log: when enabled, every call
        # to the patched set_arm_joints() records the ACTUAL post-step
        # entity qpos/qvel for this joint (read fresh from SAPIEN, not the
        # requested target), so a low-level test can confirm the fault was
        # really enforced by the physics engine rather than just requested.
        self.record_history = False
        self.history = []  # list of dicts: {step, requested_pos, requested_vel, actual_qpos, actual_qvel}
        self._step_counter = 0

    # -----------------------------------------------------------------
    # Attach / detach: monkeypatch Robot.set_arm_joints so every call
    # site in envs/_base_task.py is intercepted without modifying that
    # file.
    # -----------------------------------------------------------------
    def attach(self):
        assert not self._attached, "FaultInjector already attached"

        entity = self.robot.left_entity if self.arm_tag == "left" else self.robot.right_entity
        self._sapien_joint = entity.find_joint_by_name(self.joint_name)
        assert self._sapien_joint is not None, f"joint {self.joint_name} not found on {self.arm_tag} entity"

        if self.fault_type == "range_reduced":
            homestate = self.robot.left_homestate if self.arm_tag == "left" else self.robot.right_homestate
            home_qpos = float(homestate[self.joint_idx])
            self._apply_range_reduction(home_qpos)

        self._original_set_arm_joints = self.robot.set_arm_joints
        injector = self

        def patched_set_arm_joints(target_position, target_velocity, arm_tag):
            if arm_tag == injector.arm_tag:
                if injector.record_history:
                    actual = injector._read_current_qpos()
                    entity = injector.robot.left_entity if arm_tag == "left" else injector.robot.right_entity
                    active_joints = entity.get_active_joints()
                    qvel = entity.get_qvel()
                    arm_joints = (injector.robot.left_arm_joints if arm_tag == "left"
                                  else injector.robot.right_arm_joints)
                    actual_qvel = [float(qvel[active_joints.index(j)]) for j in arm_joints]
                    injector.history.append({
                        "step": injector._step_counter,
                        "requested_pos": float(target_position[injector.joint_idx]),
                        "requested_vel": float(target_velocity[injector.joint_idx]),
                        "actual_qpos": float(actual[injector.joint_idx]),
                        "actual_qvel": actual_qvel[injector.joint_idx],
                    })
                    injector._step_counter += 1
                if injector.fault_type is not None:
                    target_position = list(target_position)
                    target_velocity = list(target_velocity)
                    injector._apply_step_fault(target_position, target_velocity)
            return injector._original_set_arm_joints(target_position, target_velocity, arm_tag)

        self.robot.set_arm_joints = patched_set_arm_joints
        self._attached = True

    def detach(self):
        if not self._attached:
            return
        self.robot.set_arm_joints = self._original_set_arm_joints
        if self.fault_type == "range_reduced" and self._original_limits is not None:
            self._sapien_joint.set_limits([list(self._original_limits)])
        self._attached = False

    # -----------------------------------------------------------------
    # Fault mechanics
    # -----------------------------------------------------------------
    def _apply_range_reduction(self, home_qpos: float):
        lo, hi = self._normal_limits
        self._original_limits = self._sapien_joint.get_limits()[0]  # capture true pre-fault limits
        span = hi - lo
        half = 0.5 * span * self.severity
        new_lo = max(lo, home_qpos - half)
        new_hi = min(hi, home_qpos + half)
        self._sapien_joint.set_limits([[new_lo, new_hi]])

    def _apply_step_fault(self, target_position, target_velocity):
        idx = self.joint_idx

        if self.fault_type == "locked":
            if self._q_onset is None:
                self._q_onset = self._read_current_qpos()[idx]
            target_position[idx] = self._q_onset
            target_velocity[idx] = 0.0

        elif self.fault_type == "velocity_limited":
            # FIX (found via low-level verification): clipping only
            # target_velocity does essentially nothing on this robot.
            # RoboTwin's arm joints are driven by a strong POSITION PD
            # drive (stiffness=1000 per config.yml) -- when target_position
            # is far from the current position, the position-error term
            # dominates and drives real velocity toward whatever satisfies
            # that position target, regardless of what target_velocity
            # says. Verified empirically: with only target_velocity
            # clipped to 0.472 rad/s, observed |qvel| still hit 0.64 rad/s
            # and exceeded the cap on 342/967 steps.
            #
            # Correct mechanism (matches the original MuJoCo version's
            # design, which clipped the requested POSITION sequence's
            # per-step delta, not velocity): clip target_position's delta
            # from a running anchor to v_max * dt each call, and set
            # target_velocity to match that clipped rate so the drive's
            # velocity term is at least consistent with it (even though
            # the position term is what actually enforces the limit here).
            vmax = self._normal_vel_max * self.severity
            max_step_delta = vmax * self.dt
            if self._vel_anchor is None:
                self._vel_anchor = self._read_current_qpos()[idx]
            requested = target_position[idx]
            delta = float(np.clip(requested - self._vel_anchor, -max_step_delta, max_step_delta))
            new_pos = self._vel_anchor + delta
            target_position[idx] = new_pos
            target_velocity[idx] = delta / self.dt
            self._vel_anchor = new_pos

        # range_reduced needs no per-step action here: the SAPIEN joint
        # limit set in attach() is enforced by the physics engine itself
        # regardless of what target_position the planner requests.

    def _read_current_qpos(self):
        entity = self.robot.left_entity if self.arm_tag == "left" else self.robot.right_entity
        active_joints = entity.get_active_joints()
        qpos = entity.get_qpos()
        arm_joints = self.robot.left_arm_joints if self.arm_tag == "left" else self.robot.right_arm_joints
        return [qpos[active_joints.index(j)] for j in arm_joints]

    # -----------------------------------------------------------------
    # Public RPC-able getters (same shape as the MuJoCo-era FaultInjector)
    # -----------------------------------------------------------------
    def get_current_qpos(self, arm_tag: str = None):
        """Real current joint configuration for one arm (all 6 joints),
        regardless of fault state. arm_tag defaults to this injector's arm."""
        arm_tag = arm_tag or self.arm_tag
        entity = self.robot.left_entity if arm_tag == "left" else self.robot.right_entity
        active_joints = entity.get_active_joints()
        qpos = entity.get_qpos()
        arm_joints = self.robot.left_arm_joints if arm_tag == "left" else self.robot.right_arm_joints
        return [float(qpos[active_joints.index(j)]) for j in arm_joints]

    def get_fault_info(self):
        return {
            "arm_tag": self.arm_tag,
            "joint_name": self.joint_name,
            "joint_idx": self.joint_idx,
            "fault_type": self.fault_type,
            "severity": self.severity,
            "q_lock": float(self._q_onset) if self._q_onset is not None else None,
        }
