"""M5: export generated episodes (scripts/m4_generate.py .npz) to a robomimic-layout hdf5 that the
diffusion_policy pipeline reads (dp/xarm_pour_dataset.py), rendering the policy images from the
saved MuJoCo states when the episodes were generated without images.

Layout (keys follow scripts_libero/convert_libero_to_robomimic.py):
  data/demo_i/actions                      (T, 7)  float32  commanded absolute joint targets
  data/demo_i/obs/agentview_image          (T, 84, 84, 3) uint8, HWC, row 0 = top of the image
  data/demo_i/obs/robot0_eye_in_hand_image (T, 84, 84, 3) uint8
  data/demo_i/obs/robot0_eef_pos (T, 3), robot0_eef_quat (T, 4, xyzw, sign-continuous along the
      episode starting from the canonical w >= 0 form at home), robot0_joint_pos (T, 7)
  data/demo_i/states (T, nq) float32, dones, rewards
  data/demo_i attrs: num_samples, bowl_xy, grid_index, seed, n_particles, f, spill_g, success, params
Image orientation: MuJoCo's python Renderer returns row 0 = top, the same as a RealSense / OpenCV
frame, so (unlike the LIBERO files) no flip is applied; tests/test_camera.py checks it by projecting
known points with the OpenCV pinhole model of the configured camera.

  python scripts/m5_export.py --run out/m4 --out out/xarm_pour_200.hdf5 --n 200 --filter success
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h5py
import mujoco
import numpy as np

from favor_xarm.config import resolve
from favor_xarm.env import quat_continuous

_ENV = None


def _render_episode(args):
    path, cfg, rsize = args
    global _ENV
    from favor_xarm.env import PourEnv
    if _ENV is None:
        if rsize:
            cfg["cameras"]["render_size"] = list(rsize)
        _ENV = PourEnv(cfg, render=True)
    env = _ENV
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    env.set_bowl_xy(np.array(meta["bowl_xy"]))
    imgs = {"agentview_image": [], "robot0_eye_in_hand_image": []}
    if "agentview_image" in z.files:
        return path, {k: z[k] for k in imgs}
    for s in z["states"]:
        env.d.qpos[:] = s
        env.d.qvel[:] = 0
        mujoco.mj_forward(env.m, env.d)
        imgs["agentview_image"].append(env.policy_image("agentview"))
        imgs["robot0_eye_in_hand_image"].append(env.policy_image("eye_in_hand"))
    return path, {k: np.asarray(v, dtype=np.uint8) for k, v in imgs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--filter", choices=["none", "success", "quality"], default="success",
                    help="quality = f >= 0.9 and spill < 2 g (the real-robot demonstration rule)")
    ap.add_argument("--render-size", type=int, nargs=2, default=None, help="override cameras.render_size (H W)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-states", action="store_true")
    args = ap.parse_args()
    run = resolve(args.run)
    cfg = json.load(open(os.path.join(run, "config_used.json")))
    files = sorted(glob.glob(os.path.join(run, "ep_*.npz")))
    keep = []
    for p in files:
        m = json.loads(str(np.load(p)["meta"]))["metrics"]
        ok = {"none": True, "success": m["success"], "quality": m["f"] >= 0.9 and m["spill_g"] < 2.0}[args.filter]
        if ok:
            keep.append(p)
    if args.n is not None:
        keep = keep[:args.n]
    print(f"{len(files)} episodes, {len(keep)} kept ({args.filter})", flush=True)
    jobs = [(p, cfg, args.render_size) for p in keep]
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(args.workers)
        it = pool.imap(_render_episode, jobs)
    else:
        it = map(_render_episode, jobs)
    out = resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    total = 0
    with h5py.File(out, "w") as f:
        data = f.create_group("data")
        for i, (path, imgs) in enumerate(it):
            z = np.load(path)
            meta = json.loads(str(z["meta"]))
            T = z["actions"].shape[0]
            g = data.create_group(f"demo_{i}")
            g.attrs["num_samples"] = T
            g.attrs["source"] = os.path.basename(path)
            for k in ("grid_index", "seed", "n_particles"):
                g.attrs[k] = meta[k]
            g.attrs["bowl_xy"] = np.array(meta["bowl_xy"])
            for k in ("f", "spill_g", "success", "bowl_force_g"):
                g.attrs[k] = meta["metrics"][k]
            g.attrs["params"] = json.dumps(meta["params"])
            g.create_dataset("actions", data=z["actions"].astype(np.float32), compression="gzip")
            dones = np.zeros(T, dtype=np.int64); dones[-1] = 1
            rew = np.zeros(T, dtype=np.float32); rew[-1] = float(meta["metrics"]["success"])
            g.create_dataset("dones", data=dones); g.create_dataset("rewards", data=rew)
            if not args.no_states and "states" in z.files:
                g.create_dataset("states", data=z["states"], compression="gzip")
            og = g.create_group("obs")
            for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos"):
                x = z[k]
                if k == "robot0_eef_quat":            # hemisphere continuity (env.quat_continuous)
                    x = quat_continuous(x)
                og.create_dataset(k, data=x.astype(np.float32), compression="gzip")
            for k, v in imgs.items():
                assert v.shape == (T, 84, 84, 3) and v.dtype == np.uint8, (k, v.shape)
                og.create_dataset(k, data=v, compression="gzip")
            total += T
            if i % 10 == 0 or i == len(jobs) - 1:
                print(f"  demo_{i}: T={T} f={meta['metrics']['f']:.3f} elapsed {time.time() - t0:.0f}s", flush=True)
        data.attrs["total"] = total
        data.attrs["num_demos"] = len(jobs)
        data.attrs["env_args"] = json.dumps({"env_name": "favor_xarm7_pour", "type": "sim_xarm", "config": cfg})
        data.attrs["action_space"] = "absolute commanded joint targets J1..J7 (rad), 10 Hz"
        data.attrs["eef_quat_convention"] = "xyzw"
        data.attrs["image_orientation"] = "HWC uint8, row 0 = top (OpenCV/RealSense convention, not flipped)"
        data.attrs["render_size"] = json.dumps(args.render_size or cfg["cameras"]["render_size"])
    print(f"wrote {out}: {len(jobs)} demos, {total} samples, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
