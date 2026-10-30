"""M6: policy-free diagnostics on the scripted demonstration waypoints (the xArm7 counterpart of
scripts_libero/ik_select_and_layer1v2.py part B and scripts_libero/ik_convergence.py).

Waypoints: the commanded joint targets of the M4 demonstrations (re-planned from the same seeds,
bit-identical to the recorded actions), every 4th step, 25 cells x --reps. Each joint J1..J7 is
locked at its home angle. Reference frame: kinematics.tcp_frame (cup mouth by default).
For every (joint, method):
  override : the fault only overrides the joint (what B1 executes)
  W-IK pos (w_r 0.05, rho 0.1, mu 1e-6), W-IK pose (w_r 1.0, rho 0.1), Priority IK (lambda 0.01 / 0.2),
  Priority IK orientation first (lambda 0.01 / 0.2): residual position (mm), cup-axis tilt error
  (deg, the pouring-relevant part of orientation), full orientation error (deg), max |dq| of the
  healthy joints (rad), all with the rollouts' 30 iterations and 0.2 rad step clamp;
  offline safety of the corrected waypoints: joint-limit hits, robot/cup-table contact,
  self-contact, and leaving the demonstrations' cup-mouth bounding box + 5 cm (the fence rule).
Then applies the pre-registered joint selection rule (recoverable = Priority IK mean position
residual < 10 mm; proximal = the one of J1/J3 with larger override error, plus J6 and J7).
Writes reports/m6_diagnostics.{md,csv}.
"""
import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "..", "docker"))
import mujoco
import numpy as np
import torch

from favor_xarm.config import add_cli, from_args, resolve
from favor_xarm.env import grid_positions
from favor_xarm.kinematics import from_scene
from favor_xarm.scene import build_scene
from favor_xarm.scripted import ScriptedPourPlanner
from ik_priority import ik_priority
from ik_priority_rev import ik_priority_rev
from ik_redistribution import ik_redistribute, rotvec_from_matrix

# Fixed method parameters (the paper's; never tuned here).
METHODS = {
    "W-IK pos": lambda kin, q, j, c: ik_redistribute(kin, q, j, c, n_iter=30, rot_weight=0.05, reg=0.1, damping=1e-6),
    "W-IK pose": lambda kin, q, j, c: ik_redistribute(kin, q, j, c, n_iter=30, rot_weight=1.0, reg=0.1, damping=1e-6),
    "Priority IK": lambda kin, q, j, c: ik_priority(kin, q, j, c, n_iter=30, lam1=0.01, lam2=0.2),
    "Priority IK (orient. first)": lambda kin, q, j, c: ik_priority_rev(kin, q, j, c, n_iter=30, lam1=0.01, lam2=0.2),
    # Convergence check only (as scripts_libero/ik_convergence.py): same solver, 10x iterations.
    "Priority IK, 300 it (check)": lambda kin, q, j, c: ik_priority(kin, q, j, c, n_iter=300, lam1=0.01, lam2=0.2),
}


def errors(kin, qt, q):
    pt, Rt = kin.forward(qt); p, R = kin.forward(q)
    pos = (pt - p).norm(dim=-1) * 1000
    rot = torch.rad2deg(rotvec_from_matrix(Rt @ R.transpose(-1, -2)).norm(dim=-1))
    zt, z = Rt[:, :, 2], R[:, :, 2]
    tilt = torch.rad2deg(torch.acos((zt * z).sum(-1).clamp(-1, 1)))
    return pos, tilt, rot


class SafetyChecker:
    def __init__(self, cfg):
        self.h = build_scene(cfg, n_particles=0)
        self.m = self.h.model
        self.d = mujoco.MjData(self.m)
        g = self.h.geom_groups
        self.rob = np.zeros(self.m.ngeom, bool); self.rob[np.concatenate([g["robot"], g["cup"]])] = True
        self.arm = np.zeros(self.m.ngeom, bool); self.arm[g["robot"]] = True
        self.cup = np.zeros(self.m.ngeom, bool); self.cup[g["cup"]] = True
        self.tab = np.zeros(self.m.ngeom, bool); self.tab[g["table"]] = True
        lim = np.radians(np.array(cfg["robot"]["joint_limits_deg"]))
        self.lo, self.hi = lim[:, 0], lim[:, 1]

    def set_bowl(self, xy):
        self.m.body_pos[self.h.bodies["scale"], :2] = xy

    def check(self, q):
        self.d.qpos[self.h.arm_qpos] = q
        mujoco.mj_forward(self.m, self.d)
        nc = self.d.ncon
        gg = self.d.contact.geom[:nc]
        g1, g2 = gg[:, 0], gg[:, 1]
        table = bool(((self.rob[g1] & self.tab[g2]) | (self.rob[g2] & self.tab[g1])).any())
        selfc = bool(((self.arm[g1] & self.arm[g2]) | (self.arm[g1] & self.cup[g2]) | (self.arm[g2] & self.cup[g1])).any())
        limit = bool(np.any(q < self.lo - 1e-6) or np.any(q > self.hi + 1e-6))
        return table, selfc, limit, self.d.site_xpos[self.h.sites["cup_mouth"]].copy()


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--check-run", default="out/m4", help="M4 run whose recorded actions are compared to the re-plan")
    ap.add_argument("--safety-stride", type=int, default=2, help="safety check on every k-th diagnostic waypoint")
    args = ap.parse_args()
    cfg = from_args(args)
    h = build_scene(cfg, n_particles=1)
    kin = from_scene(h, cfg)
    pl = ScriptedPourPlanner(cfg, h)
    G = grid_positions(cfg)
    home = torch.tensor(cfg["robot"]["home_q"], dtype=torch.float32)

    # Waypoints (and a bit-identity check against recorded demonstrations).
    wps, bowls, n_checked, n_equal = [], [], 0, 0
    for rep in range(args.reps):
        for gi in range(len(G)):
            seed = args.base_seed + 1000 * gi + rep
            Q, _ = pl.plan(G[gi], np.random.default_rng(seed + 7))
            npz = resolve(os.path.join(args.check_run, f"ep_{rep * len(G) + gi:05d}.npz"))
            if os.path.exists(npz):
                n_checked += 1
                n_equal += int(np.array_equal(np.load(npz)["actions"], Q.astype(np.float32)))
            wps.append(Q[::args.stride]); bowls += [G[gi]] * len(Q[::args.stride])
    QT = torch.tensor(np.concatenate(wps), dtype=torch.float32)
    bowls = np.array(bowls)
    print(f"{QT.shape[0]} waypoints from {args.reps * len(G)} demonstrations; recorded actions bit-identical to "
          f"the re-plan: {n_equal}/{n_checked}", flush=True)

    sc = SafetyChecker(cfg)
    mouths = []
    for k in range(0, len(QT), args.safety_stride):
        sc.set_bowl(bowls[k]); mouths.append(sc.check(QT[k].numpy())[3])
    mouths = np.array(mouths)
    fence_lo, fence_hi = mouths.min(0) - 0.05, mouths.max(0) + 0.05

    rows = []
    for j in range(7):
        c = torch.full((len(QT),), float(home[j]))
        need = (c - QT[:, j]).abs() > 1e-6
        qt, cc = QT[need], c[need]
        q_ov = qt.clone(); q_ov[:, j] = cc
        res = {"override": (q_ov, None)}
        for name, fn in METHODS.items():
            q, info = fn(kin, qt, j, cc)
            res[name] = (q, info["dq_free"])
        for name, (q, dq) in res.items():
            pos, tilt, rot = errors(kin, qt, q)
            idx = np.arange(0, len(q), args.safety_stride)
            bidx = np.nonzero(need.numpy())[0][idx]
            fl = np.zeros((len(idx), 4), bool)
            for n, (k, b) in enumerate(zip(idx, bidx)):
                sc.set_bowl(bowls[b])
                t_, s_, l_, mo = sc.check(q[k].numpy())
                fl[n] = [t_, s_, l_, bool(np.any(mo < fence_lo) or np.any(mo > fence_hi))]
            rows.append(dict(joint=f"J{j + 1}", method=name, n=int(len(q)), bind_frac=float(need.float().mean()),
                             pos_mm=float(pos.mean()), pos_mm_p90=float(torch.quantile(pos, 0.9)),
                             pos_frac_gt10=float((pos > 10).float().mean()),
                             tilt_deg=float(tilt.mean()), tilt_deg_p90=float(torch.quantile(tilt, 0.9)),
                             rot_deg=float(rot.mean()),
                             dq_max_mean=float(dq.mean()) if dq is not None else 0.0,
                             dq_max_p95=float(torch.quantile(dq, 0.95)) if dq is not None else 0.0,
                             dq_max_max=float(dq.max()) if dq is not None else 0.0,
                             table_frac=float(fl[:, 0].mean()), self_frac=float(fl[:, 1].mean()),
                             limit_frac=float(fl[:, 2].mean()), fence_frac=float(fl[:, 3].mean())))
            r = rows[-1]
            print(f"  J{j + 1} {name:28s} pos {r['pos_mm']:7.2f} mm (p90 {r['pos_mm_p90']:7.2f})  tilt {r['tilt_deg']:6.2f} deg  "
                  f"rot {r['rot_deg']:6.2f} deg  dq {r['dq_max_mean']:.3f}/{r['dq_max_max']:.3f}  "
                  f"table {r['table_frac']:.3f} self {r['self_frac']:.3f} limit {r['limit_frac']:.3f} fence {r['fence_frac']:.3f}",
                  flush=True)

    os.makedirs(resolve("reports"), exist_ok=True)
    with open(resolve("reports/m6_diagnostics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def get(j, m, k):
        return next(r[k] for r in rows if r["joint"] == j and r["method"] == m)

    cand = ["J1", "J3", "J5", "J6", "J7"]
    recov = {j: get(j, "Priority IK", "pos_mm") < 10.0 for j in cand}
    prox = max(["J1", "J3"], key=lambda j: get(j, "override", "pos_mm"))
    sel = [j for j in (prox, "J6", "J7") if recov[j]]
    L = ["# M6: policy-free diagnostics on the scripted demonstration waypoints", "",
         f"- {QT.shape[0]} waypoints (every {args.stride}th step of {args.reps * len(G)} demonstrations, 25 cells); "
         f"each joint locked at its home angle; frame `{cfg['kinematics']['tcp_frame']}`.",
         f"- Recorded M4 actions equal the re-planned waypoints bit for bit: {n_equal}/{n_checked} checked.",
         "- IK settings are the paper's (30 iterations, 0.2 rad clamp; W-IK mu 1e-6; Priority IK lambda 0.01 / 0.2).",
         f"- Offline safety on every {args.safety_stride}nd waypoint: fraction of corrected waypoints with table contact, "
         "self contact, joint-limit hit, or cup mouth outside the demonstrations' bounding box + 5 cm (fence).", "",
         "| joint | method | pos mean (mm) | pos p90 (mm) | share > 10 mm | tilt mean (deg) | tilt p90 (deg) | rot mean (deg) | "
         "max dq healthy, mean / max (rad) | table | self | limit | fence |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['joint']} | {r['method']} | {r['pos_mm']:.2f} | {r['pos_mm_p90']:.2f} | {r['pos_frac_gt10']:.3f} | {r['tilt_deg']:.2f} | "
                 f"{r['tilt_deg_p90']:.2f} | {r['rot_deg']:.2f} | {r['dq_max_mean']:.3f} / {r['dq_max_max']:.3f} | "
                 f"{r['table_frac']:.3f} | {r['self_frac']:.3f} | {r['limit_frac']:.3f} | {r['fence_frac']:.3f} |")
    L += ["", "## Pre-registered joint selection rule", "",
          "Recoverable = Priority IK mean position residual < 10 mm. Choose: the proximal joint (J1 or J3) with the "
          "larger override error, J6, J7 (only if recoverable).", "",
          "| joint | override pos (mm) | Priority IK pos (mm) | Priority IK share > 10 mm | Priority IK 300 it pos (mm) | recoverable |",
          "|---|---|---|---|---|---|"]
    for j in cand:
        L.append(f"| {j} | {get(j, 'override', 'pos_mm'):.1f} | {get(j, 'Priority IK', 'pos_mm'):.2f} | "
                 f"{get(j, 'Priority IK', 'pos_frac_gt10'):.3f} | {get(j, 'Priority IK, 300 it (check)', 'pos_mm'):.2f} | "
                 f"{'yes' if recov[j] else 'no'} |")
    L += ["", f"Proximal candidate: **{prox}** (override {get(prox, 'override', 'pos_mm'):.1f} mm vs "
              f"{get('J3' if prox == 'J1' else 'J1', 'override', 'pos_mm'):.1f} mm). **Selected: {', '.join(sel)}**"
              + ("" if len(sel) == 3 else " (fewer than 3: see table)")]
    open(resolve("reports/m6_diagnostics.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
