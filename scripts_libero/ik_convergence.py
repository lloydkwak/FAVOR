"""
Is the W-IK / Priority IK gap a matter of structure or of convergence?  Policy-free, on
demonstration waypoints (the same targets as Layer1): 4 tasks x locked J1, J3, J5, J6, J7, 10 demos
per task, every 4th waypoint, joint locked at the demo's initial angle.

Every solver is run with the iteration limit of the rollouts (30) and with 10x that (300), same
0.2 rad step clamp. For each (solver, iterations) it reports, separately for proximal (J1, J3) and
distal (J6, J7) faults: final position residual (mean, p90, mm), final orientation residual
(mean, deg), max |dq| of the healthy joints (rad), and how many waypoints still moved by more than
1e-4 rad in the last iteration (not converged).

  W-IK  w_r in {1.0, 0.3, 0.1, 0.05, 0.01, 0.001}, rho 0, damping 1e-4 (as Priority IK's lambda_1^2)
        plus the two main settings (w_r 0.05 / 1.0 with rho 0.1, damping 1e-6)
  Priority IK  lambda_1 0.01, lambda_2 in {0.05, 0.2, 1.0}

Writes analysis_out/ik_convergence.csv and analysis_out/ik_convergence.md.
Run in the container: python /workspace/scripts_libero/ik_convergence.py
"""
import sys, csv, math, h5py, numpy as np, torch
sys.path.insert(0, "/workspace/docker")
from fault_kinematics import PandaKinematics
import ik_redistribution as ikr
from ik_priority import ik_priority
from sweep_grid_libero_range import TASKS

TASK_LIST = ["alphabet_soup", "milk", "bowl_ramekin", "bowl_stove"]
JOINTS = [1, 3, 5, 6, 7]
GROUP = {1: "proximal", 3: "proximal", 5: "J5", 6: "distal", 7: "distal"}
kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.tensor([-0.6, 0.0, 0.0]), torch.eye(3))


def load(task, n=10):
    with h5py.File(TASKS[task]["dataset"], "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:n]
        return [f[f"data/{k}/obs/robot0_joint_pos"][:] for k in keys]


def cells():
    out = []
    for t in TASK_LIST:
        qs = load(t)
        for j in JOINTS:
            ji = j - 1
            tg, con = [], []
            for q in qs:
                qt = torch.tensor(q[::4], dtype=torch.float32)
                tg.append(qt); con.append(torch.full((len(qt),), float(q[0, ji])))
            qt, c = torch.cat(tg), torch.cat(con)
            need = (c - qt[:, ji]).abs() > 1e-6
            out.append((t, j, qt[need], c[need]))
    return out


def residual(qt, q):
    pt, Rt = kin.forward(qt); p, R = kin.forward(q)
    return (pt - p).norm(dim=-1), ikr.rotvec_from_matrix(Rt @ R.transpose(-1, -2)).norm(dim=-1)


def solve(spec, qt, ji, c, iters):
    kind, a, b, damp = spec
    if kind == "wik":
        q, info = ikr.ik_redistribute(kin, qt, ji, c, n_iter=iters, rot_weight=a, reg=b, damping=damp)
        q1, _ = ikr.ik_redistribute(kin, qt, ji, c, n_iter=iters + 1, rot_weight=a, reg=b, damping=damp)
    else:
        q, info = ik_priority(kin, qt, ji, c, n_iter=iters, lam1=0.01, lam2=a)
        q1, _ = ik_priority(kin, qt, ji, c, n_iter=iters + 1, lam1=0.01, lam2=a)
    moving = (q1 - q).abs().max(dim=-1).values > 1e-4
    return q, info["dq_free"], moving


SOLVERS = {f"W-IK w_r {w:g}, rho 0": ("wik", w, 0.0, 1e-4) for w in (1.0, 0.3, 0.1, 0.05, 0.01, 0.001)}
SOLVERS.update({"W-IK pos (w_r 0.05, rho 0.1)": ("wik", 0.05, 0.1, 1e-6),
                "W-IK pose (w_r 1.0, rho 0.1)": ("wik", 1.0, 0.1, 1e-6)})
SOLVERS.update({f"Priority IK lambda_2 {l:g}": ("prio", l, None, None) for l in (0.05, 0.2, 1.0)})


def main():
    C = cells()
    print(f"{len(C)} cells, {sum(len(x[2]) for x in C)} waypoints", flush=True)
    rows = []
    for name, spec in SOLVERS.items():
        for iters in (30, 300):
            acc = {}
            for t, j, qt, c in C:
                q, dq, moving = solve(spec, qt, j - 1, c, iters)
                pe, re_ = residual(qt, q)
                g = acc.setdefault(GROUP[j], {"pe": [], "re": [], "dq": [], "mv": []})
                g["pe"].append(pe); g["re"].append(re_); g["dq"].append(dq); g["mv"].append(moving)
            for grp, g in acc.items():
                pe, re_, dq, mv = (torch.cat(g[k]) for k in ("pe", "re", "dq", "mv"))
                rows.append(dict(solver=name, iters=iters, group=grp, n=len(pe),
                                 pos_mm=1000 * pe.mean().item(), pos_mm_p90=1000 * torch.quantile(pe, 0.9).item(),
                                 rot_deg=math.degrees(re_.mean().item()), dq_max_rad=dq.mean().item(),
                                 not_converged=mv.float().mean().item()))
                r = rows[-1]
                print(f"  {name:32s} it={iters:3d} {grp:8s} pos {r['pos_mm']:6.2f} mm (p90 {r['pos_mm_p90']:6.2f}) "
                      f"rot {r['rot_deg']:5.2f} deg  dq {r['dq_max_rad']:.3f}  moving {r['not_converged']:.3f}",
                      flush=True)
    import os
    os.makedirs("/workspace/analysis_out", exist_ok=True)
    with open("/workspace/analysis_out/ik_convergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["# IK convergence and residuals (policy-free, demonstration waypoints, locked J1/J3/J5/J6/J7)", "",
         "pos / rot = final end-effector residual to the intended pose; dq = max |change| of the healthy joints; "
         "moving = share of waypoints that still change by > 1e-4 rad in one more iteration.", ""]
    for grp in ("proximal", "distal", "J5"):
        L += [f"## {grp}", "", "| solver | iters | pos mm | pos p90 mm | rot deg | dq rad | moving |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {r['solver']} | {r['iters']} | {r['pos_mm']:.2f} | {r['pos_mm_p90']:.2f} | {r['rot_deg']:.2f} | "
              f"{r['dq_max_rad']:.3f} | {r['not_converged']:.3f} |" for r in rows if r["group"] == grp]
        L.append("")
    open("/workspace/analysis_out/ik_convergence.md", "w").write("\n".join(L))
    print("IKCONV_DONE", flush=True)


if __name__ == "__main__":
    main()
