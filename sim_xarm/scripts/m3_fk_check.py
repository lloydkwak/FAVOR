"""M3: XArm7Kinematics (pytorch_kinematics + generated URDF) vs the MuJoCo scene, on 100 random
joint configurations within the official limits, for several reference frames. Also checks the
Jacobian against mj_jacSite. Pass: position < 1 mm, orientation < 0.2 deg."""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
import numpy as np
import torch

from favor_xarm.config import add_cli, from_args, resolve
from favor_xarm.kinematics import from_scene
from favor_xarm.scene import build_scene


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = from_args(args)
    h = build_scene(cfg, n_particles=0)
    m = h.model
    d = mujoco.MjData(m)
    lim = np.radians(np.array(cfg["robot"]["joint_limits_deg"]))
    rng = np.random.default_rng(args.seed)
    Q = rng.uniform(lim[:, 0], lim[:, 1], size=(args.n, 7))
    out = {}
    for frame in ("cup_mouth", "gripper_tcp", "cup_lip"):
        kin = from_scene(h, cfg, frame=frame)
        p_k, R_k = kin.forward(torch.tensor(Q, dtype=torch.float32))
        J_k = kin.jacobian(torch.tensor(Q, dtype=torch.float32)).numpy()
        pe, re, je = [], [], []
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for i, q in enumerate(Q):
            d.qpos[h.arm_qpos] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            sid = h.sites[frame]
            p = d.site_xpos[sid]; R = d.site_xmat[sid].reshape(3, 3)
            pe.append(np.linalg.norm(p - p_k[i].numpy()))
            E = R @ R_k[i].numpy().T
            re.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(E) - 1) / 2)))))
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            Jm = np.vstack([jp[:, h.arm_dof], jr[:, h.arm_dof]])
            je.append(np.abs(Jm - J_k[i]).max())
        out[frame] = dict(n=args.n, pos_err_mm_max=1000 * max(pe), pos_err_mm_mean=1000 * float(np.mean(pe)),
                          rot_err_deg_max=max(re), rot_err_deg_mean=float(np.mean(re)), jac_abs_err_max=max(je),
                          passed=bool(1000 * max(pe) < 1.0 and max(re) < 0.2))
        print(frame, json.dumps(out[frame]))
    json.dump(out, open(resolve("reports/m3_fk_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
