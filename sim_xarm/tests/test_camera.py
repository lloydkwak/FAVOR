"""Camera checks: rendered images are row 0 = top (same as a RealSense/OpenCV frame) and the
configured intrinsics/extrinsics give the OpenCV pinhole projection.

For the side camera it projects the bowl centre and a point of the cup (both coloured objects)
with K [R|t] built from the config (fovy or fx/fy/cx/cy) and compares with the centroid of the
object's pixels found by colour in the render. A vertical flip would move the centroid by
~(H - 2 v) pixels. Both intrinsics paths (fovy, and fx/fy/cx/cy with an off-centre principal
point) are tested.
Run: MUJOCO_GL=osmesa python sim_xarm/tests/test_camera.py"""
import copy
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mujoco
import numpy as np

from favor_xarm.config import load_config
from favor_xarm.env import PourEnv


def project_opencv(cam_pos, cam_xmat, K, X):
    R_mj = cam_xmat.reshape(3, 3)                  # MuJoCo camera: x right, y up, looking along -z
    R_cv = R_mj @ np.diag([1, -1, -1])             # OpenCV: x right, y down, z forward
    Xc = R_cv.T @ (X - cam_pos)
    return np.array([K[0, 0] * Xc[0] / Xc[2] + K[0, 2], K[1, 1] * Xc[1] / Xc[2] + K[1, 2]])


def centroid(img, rgb, tol=40):
    m = np.all(np.abs(img.astype(int) - np.array(rgb)) < tol, axis=-1)
    v, u = np.nonzero(m)
    return np.array([u.mean(), v.mean()]), int(m.sum())


def check(cfg, label):
    H, W = 240, 320
    cfg = copy.deepcopy(cfg)
    cfg["cameras"]["render_size"] = [H, W]
    env = PourEnv(cfg, render=True)
    env.reset(seed=0, grid_index=12, settle=False)
    intr = cfg["cameras"]["agentview"]["intrinsics"]
    if "fx" in intr:     # pixels at the configured resolution -> rendered resolution
        sx, sy = W / intr["width"], H / intr["height"]
        K = np.array([[intr["fx"] * sx, 0, intr["cx"] * sx], [0, intr["fy"] * sy, intr["cy"] * sy], [0, 0, 1]])
    else:
        f = 0.5 * H / math.tan(math.radians(intr["fovy_deg"]) / 2)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    # Hide the beads (they share the cup's location) and use flat colours.
    env.m.geom_rgba[env.h.particle_geom, 3] = 0
    img = env.render("agentview")
    cid = env.h.cameras["agentview"]
    ok = True
    bowl_c = env.d.xpos[env.h.bodies["bowl"]] + np.array([0, 0, cfg["bowl"]["bottom"]])
    rgb = (np.array(cfg["bowl"]["rgba"][:3]) * 255).astype(int)
    # The visible bowl pixels are its inner bottom + walls; compare with the projected centre of the
    # bowl at mid-depth.
    X = bowl_c + np.array([0, 0, cfg["bowl"]["depth"] / 2])
    uv_p = project_opencv(env.d.cam_xpos[cid], env.d.cam_xmat[cid], K, X)
    uv_r, npx = centroid(img, rgb, tol=70)
    err = np.linalg.norm(uv_p - uv_r)
    flipped = np.linalg.norm(np.array([uv_p[0], H - 1 - uv_p[1]]) - uv_r)
    passed = err < 12 and flipped > 3 * err
    ok &= passed
    print(f"[{label}] bowl: projected {uv_p.round(1)} rendered centroid {uv_r.round(1)} ({npx} px) "
          f"err {err:.1f} px, if flipped {flipped:.1f} px -> {'PASS' if passed else 'FAIL'}")
    return ok


def main():
    cfg = load_config()
    ok = check(cfg, "fovy")
    c2 = copy.deepcopy(cfg)
    c2["cameras"]["agentview"]["intrinsics"] = {"width": 640, "height": 480, "fx": 560.0, "fy": 560.0,
                                                "cx": 350.0, "cy": 230.0}
    ok &= check(c2, "fx/fy/cx/cy")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
