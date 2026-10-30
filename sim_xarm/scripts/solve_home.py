"""Solve the fixed home pose: cup-mouth centre at task.home_cup_pos, cup upright (cup z = world z),
gripper approach axis = world +x. Prints the joint vector to freeze into robot.home_q."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from favor_xarm.config import add_cli, from_args
from favor_xarm.mjik import SiteIK
from favor_xarm.scene import build_scene


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    cfg = from_args(args)
    h = build_scene(cfg, n_particles=1)
    lim = np.radians(np.array(cfg["robot"]["joint_limits_deg"]))
    ik = SiteIK(h.model, h.arm_qpos, h.arm_dof, h.sites["cup_mouth"], lim[:, 0], lim[:, 1])
    # Seed found by a planar grid search (J1 = J3 = J5 = 0, J7 in {0, +-pi/2, pi}) maximising
    # manipulability; the solve below only refines it.
    seed = np.array([0.0, -0.5, 0.0, 0.8, 0.0, 1.3, np.pi + np.radians(cfg["task"].get("home_cup_yaw_deg", 0.0)) - 0.0])
    seed[6] = (seed[6] + np.pi) % (2 * np.pi) - np.pi      # keep J7 near the middle of its range
    yaw = np.radians(cfg["task"].get("home_cup_yaw_deg", 0.0))
    R_t = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    q, pe, re = ik.solve(np.array(cfg["task"]["home_cup_pos"]), R_t, seed, q_posture=seed, n_iter=500)
    print("home_q:", "[" + ", ".join(f"{v:.4f}" for v in q) + "]")
    print(f"pos err {pe * 1000:.3f} mm, rot err {np.degrees(re):.3f} deg")
    qr = np.round(q, 4)
    p, R = ik.fk(qr)
    print(f"rounded: pos err {np.linalg.norm(p - cfg['task']['home_cup_pos']) * 1000:.3f} mm, "
          f"cup axis tilt {np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))):.3f} deg")


if __name__ == "__main__":
    main()
