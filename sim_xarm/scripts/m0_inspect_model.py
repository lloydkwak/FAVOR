"""M0: inspect the pinned Menagerie ufactory_xarm7 model (actuators, gains, joint limits) and
compare it with the official xArm7 specification. Writes reports/m0_model.md."""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
import numpy as np

from favor_xarm.config import add_cli, from_args, resolve

OFFICIAL_DEG = [(-360, 360), (-117, 116), (-360, 360), (-6, 225), (-360, 360), (-97, 180), (-360, 360)]


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    cfg = from_args(args)
    xml = os.path.join(resolve(cfg["assets"]["menagerie_dir"]), "ufactory_xarm7", "xarm7.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    L = [f"# M0: Menagerie ufactory_xarm7 vs official xArm7 spec", "",
         f"- MuJoCo python package: {mujoco.__version__}", f"- model: `{os.path.relpath(xml, resolve('.'))}`",
         f"- integrator: {mujoco.mjtIntegrator(m.opt.integrator).name}, timestep {m.opt.timestep} s",
         f"- nq {m.nq}, nv {m.nv}, nu {m.nu}", "",
         "## Arm joints and actuators", "",
         "| joint | MJCF range (deg) | official (deg) | diff lo / hi (deg) | damping | armature | frictionloss | actuator | kp | kv | force range (Nm) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in range(7):
        jn = f"joint{i + 1}"
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        lo, hi = np.degrees(m.jnt_range[jid])
        olo, ohi = OFFICIAL_DEG[i]
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i + 1}")
        kp = m.actuator_gainprm[aid, 0]; kv = -m.actuator_biasprm[aid, 2]
        dyn = mujoco.mjtDyn(m.actuator_dyntype[aid]).name; bias = mujoco.mjtBias(m.actuator_biastype[aid]).name
        fr = m.actuator_forcerange[aid]
        dof = m.jnt_dofadr[jid]
        L.append(f"| J{i + 1} | {lo:.1f} .. {hi:.1f} | {olo} .. {ohi} | {lo - olo:+.1f} / {hi - ohi:+.1f} | "
                 f"{m.dof_damping[dof]:g} | {m.dof_armature[dof]:g} | {m.dof_frictionloss[dof]:g} | "
                 f"general ({dyn}, bias {bias}) | {kp:g} | {kv:g} | {fr[0]:g} .. {fr[1]:g} |")
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    L += ["", f"Gripper actuator: tendon 'split', ctrlrange {m.actuator_ctrlrange[gid].tolist()}, "
              f"gain {m.actuator_gainprm[gid, 0]:g}, bias {m.actuator_biasprm[gid, :3].tolist()}", ""]
    # Gravity load at home vs actuator force limit -> steady-state tracking error without compensation.
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    L += ["## Gravity load at the Menagerie home keyframe", "",
          "| joint | gravity torque (Nm) | / force limit | steady-state error without gravity compensation (deg) |",
          "|---|---|---|---|"]
    for i in range(7):
        aid = i
        tau = d.qfrc_bias[i]
        kp = m.actuator_gainprm[aid, 0]
        L.append(f"| J{i + 1} | {tau:+.2f} | {abs(tau) / m.actuator_forcerange[aid, 1]:.2f} | {math.degrees(tau / kp):+.3f} |")
    L += ["", "## Notes", "",
          "- Actuators are position servos (`general`, affine bias = PD), no velocity limit in the model;"
          " the official 180 deg/s limit is imposed by the command layer (control.max_joint_speed_deg).",
          "- Joint ranges are overridden with the official values in the scene (robot.joint_limits_deg).",
          ]
    out = resolve("reports/m0_model.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
