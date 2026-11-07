"""M9: evaluate a trained checkpoint on the 25-cell grid with fixed seeds, optionally under joint
faults with several correction methods (paired: the same seeds and policy noise for every method).

  # no-fault gate (sim counterpart of the real-robot gate: success >= 0.75 and mean f >= 0.85)
  python scripts/m9_eval.py --ckpt <run>/checkpoints/latest.ckpt --reps 2
  # fault conditions x methods (paired)
  python scripts/m9_eval.py --ckpt ... --faults "locked:6,locked:1" --methods "B1,Priority IK,W-IK pos"

Seeds: test_start_seed + episode index (default 10000, as in the LIBERO evaluation); episode i
uses grid cell i mod 25. Training demos use seeds < 10000 and the in-training runner >= 100000.
Inference: DDIM 10 (the real-robot plan) unless --sampler ddpm. Output: <out>/episodes.csv
(one row per episode) and <out>/summary.md (per condition x method, per cell, paired counts).
Needs diffusion_policy (DIFFUSION_POLICY_ROOT) and sim_xarm/dp on the path, like training.
"""
import argparse
import csv
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
import torchvision  # noqa: F401  -- load before mujoco's GL backend (else triton segfaults at import)

from favor_xarm.config import add_cli, from_args, resolve
from favor_xarm.correction import METHODS
from favor_xarm.kinematics import from_scene
from favor_xarm.policy_rollout import RolloutPool, run_episodes, set_sampler
from favor_xarm.scene import build_scene


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=os.path.dirname(os.path.dirname(ckpt)))
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    return policy.to(device).eval(), cfg


def parse_faults(s):
    """'none' = no fault; items 'locked:J', 'range:J[:half_width]', append ':tilt' for mid-task onset."""
    out = []
    for item in (s or "none").split(","):
        item = item.strip()
        if item.lower() in ("none", "no", ""):
            out.append(None)
            continue
        parts = item.split(":")
        f = {"kind": parts[0], "joint": int(parts[1])}
        if f["kind"] not in ("locked", "range"):
            raise SystemExit(f"unknown fault kind in {item!r}")
        if f["kind"] == "range":
            f["half_width"] = float(parts[2]) if len(parts) > 2 and not parts[2].startswith("tilt") else 0.05
        if parts[-1].startswith("tilt"):
            f["onset"] = "tilt20+0.5"
        out.append(f)
    return out


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reps", type=int, default=2, help="episodes per grid cell")
    ap.add_argument("--test-start-seed", type=int, default=10000)
    ap.add_argument("--faults", default="none", help='"none" or e.g. "locked:6,locked:1,range:7:0.1,locked:7:tilt"')
    ap.add_argument("--methods", default="B1", help=f"comma-separated, from: {', '.join(METHODS)}")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    ap.add_argument("--inference-steps", type=int, default=None, help="default: 10 for ddim, as trained for ddpm")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--out", default=None, help="default: <ckpt run dir>/eval/<timestamp>")
    args = ap.parse_args()
    cfg = from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pcfg = load_policy(args.ckpt, device)
    set_sampler(policy, args.sampler, args.inference_steps)
    obs_keys = list(pcfg.task.shape_meta.obs.keys())
    n_obs, n_act = int(pcfg.n_obs_steps), int(pcfg.n_action_steps)
    methods = [m.strip() for m in args.methods.split(",")]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}; choose from {list(METHODS)}")
    faults = parse_faults(args.faults)
    n_cells = cfg["task"]["grid_n"] ** 2
    base = [dict(seed=args.test_start_seed + i, grid_index=i % n_cells) for i in range(args.reps * n_cells)]
    kin = from_scene(build_scene(cfg, n_particles=0), cfg)
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))), "eval",
                                   time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    json.dump(dict(ckpt=os.path.abspath(args.ckpt), args=vars(args), device=device,
                   num_inference_steps=int(policy.num_inference_steps),
                   scheduler=type(policy.noise_scheduler).__name__), open(os.path.join(out, "eval_config.json"), "w"), indent=1)
    pool = RolloutPool(cfg, min(args.workers, len(base)), n_obs, early_stop=not args.no_early_stop)
    rows = []
    try:
        for f in faults:
            for m in methods:
                if f is None and m != "B1" and "B1" in methods:
                    continue                      # without a fault every method is B1
                eps = [dict(e, fault=f, method=m) for e in base]
                print(f"== {('no fault' if f is None else json.dumps(f))} / {m}: {len(eps)} episodes", flush=True)
                rows += run_episodes(policy, pool, eps, obs_keys, n_act, kin=kin, log=lambda s: print("  " + s, flush=True))
    finally:
        pool.close()
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # Summary
    L = [f"# M9 evaluation: `{args.ckpt}`", "",
         f"- {len(base)} episodes per condition (25 cells x {args.reps}), seeds {args.test_start_seed}.."
         f"{args.test_start_seed + len(base) - 1}; sampler {args.sampler} {int(policy.num_inference_steps)} steps; "
         f"early stop {'off' if args.no_early_stop else 'on'}; device {device}.",
         "- success = f >= 0.8 and no safety violation (table contact, self collision, joint limit).", "",
         "| condition | method | success | mean f | mean spill (g) | safety violations | mean steps |",
         "|---|---|---|---|---|---|---|"]
    groups = {}
    for r in rows:
        groups.setdefault((r["fault"] or "no fault", r["method"]), []).append(r)
    for (c, m), rr in groups.items():
        L.append(f"| {c} | {m} | {np.mean([r['success'] for r in rr]):.2f} ({sum(r['success'] for r in rr)}/{len(rr)}) | "
                 f"{np.mean([r['f'] for r in rr]):.3f} | {np.mean([r['spill_g'] for r in rr]):.2f} | "
                 f"{sum(r['safety_violation'] for r in rr)} | {np.mean([r['steps'] for r in rr]):.0f} |")
    if ("no fault", "B1") in groups:
        rr = groups[("no fault", "B1")]
        s, fm = np.mean([r["success"] for r in rr]), np.mean([r["f"] for r in rr])
        L += ["", f"**No-fault gate (sim counterpart of the pre-registered real gate: success >= 0.75 and mean f >= 0.85): "
                  f"{'PASS' if (s >= 0.75 and fm >= 0.85) else 'FAIL'}** (success {s:.2f}, mean f {fm:.3f})"]
        L += ["", "Per cell, no fault (success rate; rows x near to far, columns y):", ""]
        G = np.zeros((n_cells,)); N = np.zeros((n_cells,))
        for r in rr:
            G[r["grid_index"]] += r["success"]; N[r["grid_index"]] += 1
        n = cfg["task"]["grid_n"]
        L += ["| row \\ col | " + " | ".join(str(c) for c in range(n)) + " |", "|---" * (n + 1) + "|"]
        for i in range(n):
            L.append(f"| {i} | " + " | ".join(f"{G[i * n + j] / N[i * n + j]:.2f}" if N[i * n + j] else "-" for j in range(n)) + " |")
    if "Priority IK" in methods and len(methods) > 1:
        L += ["", "Paired comparison per fault (Priority IK only success : other only success):", ""]
        for f in faults:
            if f is None:
                continue
            c = next(r["fault"] for r in rows if r["fault"] and r["fault"].endswith(f"J{f['joint']}" + (" tilt20+0.5" if f.get("onset") else "")))
            p = {r["seed"]: r["success"] for r in groups.get((c, "Priority IK"), [])}
            for m in methods:
                if m == "Priority IK":
                    continue
                o = {r["seed"]: r["success"] for r in groups.get((c, m), [])}
                a = sum(1 for k in p if p[k] and not o.get(k, 0)); b = sum(1 for k in p if o.get(k, 0) and not p[k])
                L.append(f"- {c}: Priority IK vs {m}: {a}:{b}")
    txt = "\n".join(L) + "\n"
    open(os.path.join(out, "summary.md"), "w").write(txt)
    print(txt)
    print("wrote", out)


if __name__ == "__main__":
    main()
