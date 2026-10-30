"""M8: roll out a trained diffusion_policy checkpoint in the pouring env, optionally under a joint
fault with a post-hoc correction method (B1 / W-IK pos / W-IK pose / Priority IK / orientation
first). Same structure as the LIBERO runs: n_obs 2, predict 16, execute 8; the policy noise is
seeded by (episode seed, call index) so every method sees the same samples until their states
diverge.

  python scripts/m8_rollout.py --ckpt <...>/checkpoints/latest.ckpt --episodes 3 --cells 12,0,24
  python scripts/m8_rollout.py --ckpt ... --fault-joint 6 --fault locked --method "Priority IK"

Needs diffusion_policy (DIFFUSION_POLICY_ROOT) and sim_xarm/dp on the path, as for training.
"""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path[:0] = [SIM, os.path.join(SIM, "dp"),
                os.environ.get("DIFFUSION_POLICY_ROOT", os.path.join(SIM, "..", "diffusion_policy"))]
import dill
import hydra
import numpy as np
import torch
import torchvision  # noqa: F401  -- must load (torch._dynamo / triton) before mujoco's GL backend: else segfault

from favor_xarm.config import add_cli, from_args, resolve
from favor_xarm.correction import METHODS, correct_chunk
from favor_xarm.env import PourEnv
from favor_xarm.faults import FaultLayer
from favor_xarm.kinematics import from_scene


def load_policy(ckpt, device, inference_steps=None):
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    ws = cls(cfg, output_dir=os.path.dirname(os.path.dirname(ckpt)))
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    policy.to(device).eval()
    if inference_steps is not None:
        policy.num_inference_steps = inference_steps
    return policy, cfg


def obs_to_policy(hist, keys, device):
    out = {}
    for k in keys:
        x = np.stack([h[k] for h in hist])
        if x.ndim == 4:                                           # T, H, W, C uint8 -> T, C, H, W float
            x = np.moveaxis(x, -1, 1).astype(np.float32) / 255.0
        out[k] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(device)
    return out


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--cells", type=str, default="12")
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--inference-steps", type=int, default=None, help="DDPM steps (default: as trained)")
    ap.add_argument("--fault-joint", type=int, default=None, help="1..7")
    ap.add_argument("--fault", choices=["locked", "range"], default="locked")
    ap.add_argument("--onset", default=None, choices=[None, "tilt20+0.5"])
    ap.add_argument("--method", default="B1", choices=list(METHODS))
    ap.add_argument("--out", default="out/m8_rollouts.jsonl")
    args = ap.parse_args()
    cfg = from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pcfg = load_policy(args.ckpt, device, args.inference_steps)
    keys = list(pcfg.task.shape_meta.obs.keys())
    n_obs, n_act = pcfg.n_obs_steps, pcfg.n_action_steps
    env = PourEnv(cfg, render=True)
    kin = from_scene(env.h, cfg)
    if args.fault_joint is not None:
        env.fault = FaultLayer(args.fault_joint - 1, args.fault, half_width=0.05, onset=args.onset)
    cells = [int(c) for c in args.cells.split(",")]
    max_steps = args.max_steps or cfg["task"]["max_steps"]
    out = resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for e in range(args.episodes):
        gi = cells[e % len(cells)]
        seed = args.base_seed + e
        t0 = time.time()
        obs = env.reset(seed=seed, grid_index=gi)
        hist = collections.deque([obs] * n_obs, maxlen=n_obs)
        step, call, logs, t_inf = 0, 0, [], 0.0
        while step < max_steps:
            torch.manual_seed(seed * 100003 + call)
            ti = time.time()
            with torch.no_grad():
                act = policy.predict_action(obs_to_policy(hist, keys, device))["action"][0].cpu()
            t_inf += time.time() - ti
            call += 1
            info = env.fault.get_fault_info() if env.fault is not None else None
            act, lg = correct_chunk(kin, act[:n_act], info, args.method)
            if lg:
                logs.append(lg)
            for q in act.numpy():
                obs = env.step(q)
                hist.append(obs)
                step += 1
                if step >= max_steps:
                    break
        env.hold(cfg["task"]["settle_after"])
        m = env.metrics()
        row = dict(episode=e, seed=seed, grid_index=gi, method=args.method,
                   fault=None if args.fault_joint is None else f"{args.fault} J{args.fault_joint}" + (f" {args.onset}" if args.onset else ""),
                   steps=step, calls=call, wall_s=time.time() - t0, inference_s_per_call=t_inf / max(call, 1),
                   n_corrected_waypoints=sum(l["n"] for l in logs),
                   corr_pos_after_mm=float(np.mean([l["pos_after_mm"] for l in logs])) if logs else None,
                   corr_dq=float(np.mean([l["dq"] for l in logs])) if logs else None,
                   **{k: m[k] for k in ("f", "spill_g", "cup_g", "success", "safety_violation", "max_tilt_deg")},
                   safety_first=str(m["safety_first"]))
        print(json.dumps(row, default=float), flush=True)
        open(out, "a").write(json.dumps(row, default=float) + "\n")


if __name__ == "__main__":
    main()
