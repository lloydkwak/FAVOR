"""M4 report: per-grid-cell success rate, mean transfer fraction, spill and duration from
summary.csv. Pass: overall success >= 95 %, mean f >= 0.9, every demonstration <= 30 s."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from favor_xarm.config import load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = resolve(args.run)
    rows = list(csv.DictReader(open(os.path.join(run, "summary.csv"))))
    cfg = load_config()
    n = cfg["task"]["grid_n"]
    from favor_xarm.env import grid_positions
    G = grid_positions(cfg)
    L = [f"# M4: scripted demonstrations ({len(rows)} episodes, `{args.run}`)", ""]
    f = np.array([float(r["f"]) for r in rows]); s = np.array([float(r["spill_g"]) for r in rows])
    succ = np.array([int(r["success"]) for r in rows]); dur = np.array([float(r["duration_s"]) for r in rows])
    viol = np.array([int(r["safety_violation"]) for r in rows])
    wall = np.array([float(r["wall_s"]) for r in rows])
    L += [f"- success (f >= 0.8, no safety violation): **{succ.mean():.3f}** ({succ.sum()}/{len(rows)})",
          f"- mean f: **{f.mean():.3f}** (min {f.min():.3f}); mean spill {s.mean():.2f} g (max {s.max():.2f} g)",
          f"- demonstrations with f >= 0.9 and spill < 2 g (real-robot quality rule): {((f >= 0.9) & (s < 2)).mean():.3f}",
          f"- duration {dur.min():.1f}-{dur.max():.1f} s (limit 30 s); safety violations {viol.sum()}",
          f"- wall time per episode (this machine, physics only): mean {wall.mean():.0f} s", "",
          "Per cell (row = x from near to far, column = y from right to left; cell index = row * 5 + col):", "",
          "| cell | bowl x, y (m) | n | success | mean f | min f | mean spill (g) | max duration (s) |",
          "|---|---|---|---|---|---|---|---|"]
    worst = 1.0
    for gi in range(n * n):
        rr = [r for r in rows if int(r["grid_index"]) == gi]
        if not rr:
            continue
        ff = np.array([float(r["f"]) for r in rr]); sc = np.array([int(r["success"]) for r in rr])
        ss = np.array([float(r["spill_g"]) for r in rr]); dd = np.array([float(r["duration_s"]) for r in rr])
        worst = min(worst, sc.mean())
        L.append(f"| {gi} | {G[gi][0]:.2f}, {G[gi][1]:.2f} | {len(rr)} | {sc.mean():.2f} | {ff.mean():.3f} | {ff.min():.3f} | "
                 f"{ss.mean():.2f} | {dd.max():.1f} |")
    L += ["", "Success rate grid (rows x, columns y):", "", "| x \\ y | " + " | ".join(f"{G[c][1]:.2f}" for c in range(n)) + " |",
          "|---" * (n + 1) + "|"]
    for r_ in range(n):
        cells = []
        for c in range(n):
            rr = [r for r in rows if int(r["grid_index"]) == r_ * n + c]
            cells.append(f"{np.mean([int(r['success']) for r in rr]):.1f}" if rr else "-")
        L.append(f"| {G[r_ * n][0]:.2f} | " + " | ".join(cells) + " |")
    passed = succ.mean() >= 0.95 and f.mean() >= 0.9 and dur.max() <= 30.0
    L += ["", f"Lowest per-cell success: {worst:.2f}. **M4 criterion (>= 95 %, mean f >= 0.9, <= 30 s): "
              f"{'PASS' if passed else 'FAIL'}**"]
    txt = "\n".join(L) + "\n"
    open(args.out or resolve("reports/m4_demos.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
