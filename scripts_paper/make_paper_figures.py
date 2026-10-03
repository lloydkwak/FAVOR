"""
Paper figures and tables from the raw sweep results.

Reads results/<sweep dirs>/*.json (per-episode success, same seeds for all methods)
and writes:
  paper/figs/*.pdf, *.png         figures
  paper/tables/*.tex, *.csv        tables (booktabs LaTeX + CSV)
  paper/tables/all_conditions.csv  one row per (level, task, joint) with every method's score

Usage (host, from repo root):  python scripts_paper/make_paper_figures.py [--results results] [--out paper]
Only json / csv / numpy / matplotlib are needed. Missing sweeps are skipped, never faked.
"""
import argparse, csv, json, os
from math import comb, sqrt

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["alphabet_soup", "milk", "bowl_ramekin", "bowl_stove"]
TASK_LABEL = {"alphabet_soup": "Soup", "milk": "Milk", "bowl_ramekin": "Bowl-Ramekin", "bowl_stove": "Bowl-Stove"}
JOINTS = list(range(1, 8))
LEVELS = ["mild", "moderate", "severe", "locked"]

# method key -> (label, locked dir, range dir)
METHODS = {
    "b1":       ("B1 (no intervention)", "libero_fault_sweep_locked_b1",       "libero_fault_sweep_range_b1"),
    "random_n": ("Random-N",             "libero_fault_sweep_phase2_random_n", "libero_fault_sweep_range_random_n"),
    "select":   ("Select ($\\epsilon$-cert.)",     "libero_fault_sweep_phase2_select",   "libero_fault_sweep_range_select"),
    "eci":      ("E-C-I (projection)",   "libero_fault_sweep_eci",             "libero_fault_sweep_range_eci"),
    "rg":       ("RG-DDPM (guidance)",   "libero_fault_sweep_locked_rg",       "libero_fault_sweep_range_rg"),
    "pos":      ("W-IK pos",             "libero_fault_sweep_locked_ik",       "libero_fault_sweep_range_ik"),
    "pose":     ("W-IK pose",            "libero_fault_sweep_locked_ik_pose",  "libero_fault_sweep_range_ik_pose"),
    "prio":     ("Priority IK",   "libero_fault_sweep_locked_prio",     "libero_fault_sweep_range_prio"),
}
SHORT = {"b1": "B1", "random_n": "Rand-N", "select": "Select", "eci": "E-C-I", "rg": "RG-DDPM",
         "pos": "W-IK pos", "pose": "W-IK pose", "prio": "\\textbf{Prio-IK}"}
COLOR = {"b1": "#9E9C9C", "random_n": "#CFCFCF", "select": "#B8C4D9", "eci": "#163A78", "rg": "#5B7DB8",
         "pos": "#E39A9E", "pose": "#F2C4C6", "prio": "#B4131C", "best": "#5A5A5A"}
MS = {  # robustness to misspecified fault knowledge (Priority IK); table order
    "rs050":  ("window $\\times$0.5",  "range", "moderate", "libero_fault_sweep_range_ms_rs050"),
    "rs150":  ("window $\\times$1.5",  "range", "moderate", "libero_fault_sweep_range_ms_rs150"),
    "lop001": ("lock +0.01 rad", "locked", "locked", "libero_fault_sweep_locked_ms_lop001"),
    "lop002": ("lock +0.02 rad", "locked", "locked", "libero_fault_sweep_locked_ms_lop002"),
    "lop005": ("lock +0.05 rad", "locked", "locked", "libero_fault_sweep_locked_ms_lop005"),
    "lop010": ("lock +0.1 rad", "locked", "locked", "libero_fault_sweep_locked_ms_lop010"),
    "lom010": ("lock $-$0.1 rad", "locked", "locked", "libero_fault_sweep_locked_ms_lom010"),
}
FIG_MS = {"rs050": "window\n$\\times$0.5", "rs150": "window\n$\\times$1.5",   # bar figure: main variants only
          "lop010": "lock\n+0.1 rad", "lom010": "lock\n$-$0.1 rad"}             # (angle sweep -> Fig 6)
N50_METHODS = ["b1", "ik", "ik_pose", "prio"]
N50_LABEL = {"b1": "B1", "ik": "W-IK pos", "ik_pose": "W-IK pose", "prio": "Priority IK"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def fname(level, task, j):
    if level == "locked":
        return f"{task}_robot0_joint{j}_locked_na.json"
    return f"{task}_robot0_joint{j}_range_reduced_{level}.json"


def load(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    pe = d.get("per_episode")
    pe = {str(k): float(v) for k, v in pe.items()} if pe else None
    if pe:
        score = sum(pe.values()) / len(pe)          # per-episode mean is the ground truth
    else:                                            # older files: value stored under the method key
        keys = ("b1", "eci", "select", "random_n", "ik", "ik_pose", "prio", "rg", "score", "mean_score")
        score = next((float(d[k]) for k in keys if isinstance(d.get(k), (int, float))), None)
        if score is None:
            score = next((float(v) for k, v in d.items() if k.startswith("ms_") and isinstance(v, (int, float))), None)
    return {"score": score, "pe": pe, "raw": d}


def collect(R):
    data = {}
    for m, (_, dl, dr) in METHODS.items():
        for level in LEVELS:
            d = dl if level == "locked" else dr
            for t in TASKS:
                for j in JOINTS:
                    rec = load(os.path.join(R, d, fname(level, t, j)))
                    if rec is not None and rec["score"] is not None:
                        data[(level, t, j, m)] = rec
    return data


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def paired(P, Q):
    b = sum(1 for k in P if P[k] > .5 and Q.get(k, 0.0) <= .5)
    c = sum(1 for k in P if P[k] <= .5 and Q.get(k, 0.0) > .5)
    return b, c


def mean_ci(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return np.nan, np.nan
    se = v.std(ddof=1) / sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), 1.96 * se


def fmt_p(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def best_of(data, level, t, j):
    a, b = data.get((level, t, j, "pos")), data.get((level, t, j, "pose"))
    if a is None or b is None:
        return None
    return a if a["score"] >= b["score"] else b


def write_table(out, name, header, rows, caption, label, colspec=None):
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    with open(os.path.join(out, "tables", name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    colspec = colspec or ("l" + "c" * (len(header) - 1))
    with open(os.path.join(out, "tables", name + ".tex"), "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n")
        f.write("\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{" + colspec + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def savefig(fig, out, name):
    os.makedirs(os.path.join(out, "figs"), exist_ok=True)
    for ext in ("pdf", "png"):
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(os.path.join(out, "figs", f"{name}.{ext}"), bbox_inches="tight", dpi=300, metadata=meta)
    plt.close(fig)


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float); o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
        for u in np.unique(v):                      # average ranks for ties
            m = v == u; r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def layer1(data, path, OUT, report):
    """Layer-1 kinematic analysis vs. rollouts: does the policy-free EE error predict B1 success,
    and does the residual after IK predict what post-hoc retargeting can recover?"""
    if not os.path.exists(path):
        print(f"  layer1: {path} not found -> skipped"); return
    rows = list(csv.DictReader(open(path)))
    x1, y1, x2, y2, lv = [], [], [], [], []
    for r in rows:
        j = int(r["joint"][-1]); key = (r["level"], r["task"], j)
        B, P = data.get(key + ("b1",)), data.get(key + ("prio",))
        if B is None or P is None:
            continue
        x1.append(float(r["override_pos_mm"])); y1.append(B["score"])
        x2.append(float(r["ik_pos_mm"])); y2.append(P["score"]); lv.append(r["level"])
    if len(x1) < 10:
        print("  layer1: too few matched conditions -> skipped"); return
    rho1, rho2 = spearman(x1, y1), spearman(x2, y2)
    lvcol = {"mild": "#8FA6CC", "moderate": "#163A78", "severe": "#E39A9E", "locked": "#B4131C"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for ax, xs, ys, xl, yl, rho in [(axs[0], x1, y1, "EE error if the fault just overrides the joint [mm]", "B1 success", rho1),
                                    (axs[1], x2, y2, "EE error left after IK retargeting [mm]", "Priority IK success", rho2)]:
        for L in LEVELS:
            idx = [i for i, l in enumerate(lv) if l == L]
            ax.scatter([max(xs[i], 0.05) for i in idx], [ys[i] for i in idx], s=9, color=lvcol[L], label=L.capitalize(),
                       edgecolors="none", alpha=0.85)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Spearman $\\rho$ = {rho:.2f}  (n={len(xs)})")
        ax.yaxis.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()   # one shared legend above the panels, clear of the data
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(l), frameon=False, fontsize=7,
               markerscale=1.6, handletextpad=0.2, columnspacing=1.2, title="Fault level", title_fontsize=7)
    fig.tight_layout()
    savefig(fig, OUT, "fig_layer1"); report.append("fig_layer1")
    write_table(OUT, "tab_layer1", ["Predictor", "Outcome", "n", "Spearman $\\rho$"],
                [["override EE error", "B1 success", len(x1), f"{rho1:.2f}"],
                 ["residual EE error after IK", "Priority IK success", len(x2), f"{rho2:.2f}"]],
                "Policy-free kinematic analysis (Layer 1) vs. closed-loop rollouts.", "tab:layer1")
    report.append("tab_layer1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--layer1", default="analysis_out/layer1_v2.csv")
    a = ap.parse_args()
    R, OUT = a.results, a.out
    data = collect(R)
    have = sorted({m for (_, _, _, m) in data})
    print("methods found:", have)
    report = []

    rows = []
    for level in LEVELS:
        for t in TASKS:
            for j in JOINTS:
                rows.append([level, t, j] + [("" if (level, t, j, m) not in data else f"{data[(level, t, j, m)]['score']:.3f}")
                                             for m in METHODS])
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    with open(os.path.join(OUT, "tables", "all_conditions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["level", "task", "joint"] + list(METHODS)); w.writerows(rows)

    means = {}
    for level in LEVELS:
        for m in METHODS:
            v = [data[(level, t, j, m)]["score"] for t in TASKS for j in JOINTS if (level, t, j, m) in data]
            if v:
                means[(level, m)] = (*mean_ci(v), len(v))
        vb = [best_of(data, level, t, j)["score"] for t in TASKS for j in JOINTS if best_of(data, level, t, j)]
        if vb:
            means[(level, "best")] = (*mean_ci(vb), len(vb))

    # Table 1: main results
    cols = [m for m in METHODS if any((lv, m) in means for lv in LEVELS)]
    header = ["Fault level"] + [SHORT[m] for m in cols] + ["Best W-IK$^\\dagger$"]
    trows = []
    for level in LEVELS:
        row = [level]
        vals = {m: means[(level, m)][0] for m in cols if (level, m) in means}
        top = max(vals.values()) if vals else None
        for m in cols:
            if (level, m) in means:
                mu, _, n = means[(level, m)]
                s = f"{mu:.2f}" + ("" if n == 28 else f"$^{{({n})}}$")
                row.append(f"\\textbf{{{s}}}" if top is not None and abs(mu - top) < 1e-9 else s)
            else:
                row.append("--")
        row.append(f"{means[(level, 'best')][0]:.2f}" if (level, "best") in means else "--")
        trows.append(row)
    allrow = ["all (112)"]
    for m in cols + ["best"]:
        if all((lv, m) in means and means[(lv, m)][2] == 28 for lv in LEVELS):
            allrow.append(f"{np.mean([means[(lv, m)][0] for lv in LEVELS]):.2f}")
        else:
            allrow.append("--")
    trows.append(allrow)
    write_table(OUT, "tab_main", header, trows,
                "Mean success rate over 28 conditions (4 tasks $\\times$ 7 joints, $n{=}20$ episodes, identical seeds) "
                "per fault level. $^\\dagger$Best W-IK picks, per condition, the better W-IK setting after seeing the "
                "results. Superscripts give the number of conditions when fewer than 28 were run.", "tab:main")
    report.append("tab_main")

    # Table 2: paired tests Priority IK vs others
    if any(k[3] == "prio" for k in data):
        comp = [m for m in ["b1", "eci", "rg", "pos", "pose"] if m in have] + ["best"]
        header = ["Fault level"] + [("vs " + (SHORT[m] if m != "best" else "Best W-IK$^\\dagger$")) for m in comp]
        trows, tot = [], {m: [0, 0] for m in comp}
        for level in LEVELS:
            row = [level]
            for m in comp:
                b = c = 0; ok = False
                for t in TASKS:
                    for j in JOINTS:
                        P = data.get((level, t, j, "prio"))
                        Q = best_of(data, level, t, j) if m == "best" else data.get((level, t, j, m))
                        if P and Q and P["pe"] and Q["pe"]:
                            bb, cc = paired(P["pe"], Q["pe"]); b += bb; c += cc; ok = True
                if ok:
                    tot[m][0] += b; tot[m][1] += c
                    p = mcnemar(b, c); cell = f"{b}:{c} ({fmt_p(p)})"
                    row.append(f"\\textbf{{{cell}}}" if (p < .05 and b > c) else cell)
                else:
                    row.append("--")
            trows.append(row)
        allr = ["all"]
        for m, (b, c) in tot.items():
            p = mcnemar(b, c); cell = f"{b}:{c} ({fmt_p(p)})"
            allr.append(f"\\textbf{{{cell}}}" if (p < .05 and b > c) else cell)
        trows.append(allr)
        write_table(OUT, "tab_paired", header, trows,
                    "Episode-level paired comparison of Priority IK against each method (same seeds). Each cell: "
                    "episodes won only by Priority IK : episodes won only by the other method (exact McNemar $p$). "
                    "Bold: Priority IK significantly better ($p<0.05$).", "tab:paired")
        report.append("tab_paired")

    # Fig 1: method comparison per fault level
    order = [m for m in ["b1", "random_n", "select", "eci", "rg", "pos", "pose", "prio"] if m in have]
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    W = 0.8 / len(order)
    for i, m in enumerate(order):
        xs, ys, es = [], [], []
        for li, level in enumerate(LEVELS):
            if (level, m) in means and means[(level, m)][2] == 28:
                mu, ci, n = means[(level, m)]
                xs.append(li + (i - (len(order) - 1) / 2) * W); ys.append(mu); es.append(ci)
        if not xs:
            continue
        ax.bar(xs, ys, W * 0.95, yerr=es, color=COLOR[m], label=METHODS[m][0], error_kw=dict(lw=0.6, capsize=1.2),
               edgecolor="black" if m == "prio" else "none", linewidth=0.5)
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(["Mild", "Moderate", "Severe", "Locked"])
    ax.set_ylabel("Success rate"); ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.18))
    savefig(fig, OUT, "fig_methods_by_level"); report.append("fig_methods_by_level")

    # Fig 2: per-joint locked
    jm = [m for m in ["b1", "eci", "pos", "pose", "prio"] if m in have]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    W = 0.8 / len(jm)
    for i, m in enumerate(jm):
        ys = [np.mean([data[("locked", t, j, m)]["score"] for t in TASKS if ("locked", t, j, m) in data] or [np.nan])
              for j in JOINTS]
        ax.bar(np.arange(7) + (i - (len(jm) - 1) / 2) * W, ys, W * 0.95, color=COLOR[m], label=METHODS[m][0],
               edgecolor="black" if m == "prio" else "none", linewidth=0.5)
    for j in JOINTS:
        allv = [data[("locked", t, j, m)]["score"] for m in jm for t in TASKS if ("locked", t, j, m) in data]
        if allv and max(allv) < 0.15:
            ax.text(j - 1, 0.03, "all\n≈0", ha="center", va="bottom", fontsize=5.5, color="#777777")
    ax.set_xticks(range(7)); ax.set_xticklabels([f"J{j}" for j in JOINTS])
    ax.set_ylabel("Success rate (locked)"); ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.3), columnspacing=0.8, handlelength=1.0)
    savefig(fig, OUT, "fig_locked_by_joint"); report.append("fig_locked_by_joint")

    # Fig 3: Priority IK minus best(pos,pose) heatmap
    if "prio" in have:
        fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.6), sharey=True)
        for ax, level in zip(axs, LEVELS):
            M = np.full((len(TASKS), 7), np.nan)
            for ti, t in enumerate(TASKS):
                for j in JOINTS:
                    P, B = data.get((level, t, j, "prio")), best_of(data, level, t, j)
                    if P and B:
                        M[ti, j - 1] = P["score"] - B["score"]
            im = ax.pcolormesh(np.arange(8) - 0.5, np.arange(len(TASKS) + 1) - 0.5, np.ma.masked_invalid(M),
                               cmap="RdBu_r", vmin=-0.6, vmax=0.6, edgecolors="white", linewidth=0.4)
            ax.set_xlim(-0.5, 6.5); ax.set_ylim(len(TASKS) - 0.5, -0.5)
            ax.set_title(level.capitalize()); ax.set_xticks(range(7)); ax.set_xticklabels([f"J{j}" for j in JOINTS])
            ax.set_yticks(range(len(TASKS))); ax.set_yticklabels([TASK_LABEL[t] for t in TASKS])
            for (r, c), v in np.ndenumerate(M):
                if not np.isnan(v) and abs(v) >= 0.15:
                    ax.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=5,
                            color="white" if abs(v) > 0.35 else "black")
        cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01)
        cb.set_label("Priority IK - best(pos,pose)", fontsize=6); cb.ax.tick_params(labelsize=6)
        savefig(fig, OUT, "fig_prio_minus_best_heatmap"); report.append("fig_prio_minus_best_heatmap")

    # Fig 4: severity curves
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    for m in [x for x in ["b1", "eci", "rg", "pos", "pose", "prio"] if x in have]:
        ys = [means[(lv, m)][0] if (lv, m) in means and means[(lv, m)][2] == 28 else np.nan for lv in LEVELS]
        if all(np.isnan(ys)):
            continue
        ax.plot(range(4), ys, marker="o", ms=3, lw=1.6 if m == "prio" else 1.0, color=COLOR[m], label=METHODS[m][0])
    if all((lv, "best") in means for lv in LEVELS):
        ax.plot(range(4), [means[(lv, "best")][0] for lv in LEVELS], ls="--", lw=0.9, color=COLOR["best"],
                label="Best W-IK (per condition)")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Mild", "Moderate", "Severe", "Locked"])
    ax.set_ylabel("Mean success rate"); ax.set_ylim(0, 0.9)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=6)
    savefig(fig, OUT, "fig_severity_curves"); report.append("fig_severity_curves")

    # Table 3 / Fig 5: misspecified fault knowledge
    rrows, bars = [], []
    for tag, (lab, kind, level, d) in MS.items():
        b = c = 0; ms_s, ex_s, n = [], [], 0
        for t in TASKS:
            for j in JOINTS:
                Q = load(os.path.join(R, d, fname(level, t, j)))
                P = data.get((level, t, j, "prio"))
                if Q is None or P is None or Q["score"] is None:
                    continue
                n += 1; ms_s.append(Q["score"]); ex_s.append(P["score"])
                if Q["pe"] and P["pe"]:
                    bb, cc = paired(Q["pe"], P["pe"]); b += bb; c += cc
        if n and n < 20:
            print(f"  misspec {tag}: {n}/20 conditions done -> skipped until complete")
        if n == 20:
            rrows.append([lab, level, n, f"{np.mean(ex_s):.2f}", f"{np.mean(ms_s):.2f}",
                          f"{np.mean(ms_s) - np.mean(ex_s):+.2f}", f"{b}:{c} ({fmt_p(mcnemar(b, c))})"])
            if tag in FIG_MS:
                bars.append((FIG_MS[tag], np.mean(ex_s), np.mean(ms_s)))
    if rrows:
        write_table(OUT, "tab_misspec",
                    ["Misspecification", "Level", "Cond.", "Exact", "Misspec.", "$\\Delta$", "paired (misspec:exact)"],
                    rrows, "Priority IK with misspecified fault knowledge (joints 1,3,5,6,7). Exact = same conditions "
                           "with the true fault parameters.", "tab:misspec")
        fig, ax = plt.subplots(figsize=(3.5, 1.8))
        x = np.arange(len(bars))
        ax.bar(x - 0.2, [b[1] for b in bars], 0.38, color=COLOR["prio"], label="exact")
        ax.bar(x + 0.2, [b[2] for b in bars], 0.38, color=COLOR["pos"], label="misspecified")
        ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars]); ax.set_ylim(0, 1); ax.set_ylabel("Success rate")
        ax.legend(frameon=False, ncol=2, loc="upper right"); ax.yaxis.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
        ax.set_title("Priority IK, 20 conditions each", fontsize=7)
        savefig(fig, OUT, "fig_misspec"); report.append("tab_misspec, fig_misspec")

    # Fig 6: sensitivity to the lock-angle error (Priority IK, joints 1,3,5,6,7, locked)
    J5 = [1, 3, 5, 6, 7]
    def mean_over(dirname, key, joints=None):
        v = []
        for t in TASKS:
            for j in (joints or J5):
                r = load(os.path.join(R, dirname, fname("locked", t, j))) if dirname else data.get(("locked", t, j, key))
                if r is None or r["score"] is None:
                    return None
                v.append(r["score"])
        return float(np.mean(v))
    GROUPS = [("all (J1,3,5,6,7)", J5, "-", 1.6, 3.5), ("proximal (J1,3)", [1, 3], "--", 1.0, 2.5),
              ("distal (J5,6,7)", [5, 6, 7], ":", 1.0, 2.5)]
    curves = {}
    for g, js, _, _, _ in GROUPS:
        pts_g = [(0.0, mean_over(None, "prio", js))]
        for off, tag in [(0.01, "lop001"), (0.02, "lop002"), (0.05, "lop005"), (0.10, "lop010")]:
            m = mean_over(MS[tag][3], None, js)
            if m is not None:
                pts_g.append((off, m))
        curves[g] = pts_g
    pts = curves[GROUPS[0][0]]
    b1 = mean_over(None, "b1")
    if len(pts) >= 3 and pts[0][1] is not None and b1 is not None:
        fig, ax = plt.subplots(figsize=(3.5, 1.9))
        for g, _, ls, lw, ms in GROUPS:
            c = curves[g]
            ax.plot([p[0] for p in c], [p[1] for p in c], marker="o", ms=ms, lw=lw, ls=ls, color=COLOR["prio"],
                    alpha=1.0 if lw > 1.2 else 0.75, label=f"Priority IK, {g}")
        ax.set_xticks([0, 0.01, 0.02, 0.05, 0.10]); ax.set_xticklabels(["0", ".01", ".02", ".05", ".10"])
        ax.axhline(b1, ls="--", lw=0.9, color=COLOR["b1"], label="B1 (no intervention)")
        ax.set_xlabel("error in the assumed lock angle [rad]"); ax.set_ylabel("Success rate (locked)")
        ax.set_ylim(0, 1); ax.yaxis.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=5.5, loc="upper right")
        savefig(fig, OUT, "fig_lock_angle_sensitivity"); report.append("fig_lock_angle_sensitivity")
    else:
        print("  lock-angle sensitivity: need >=3 points -> skipped")

    # Table 4: n=50 fresh-seed replication
    d50 = os.path.join(R, "libero_confirm_n50_fresh")
    if os.path.isdir(d50):
        conds = sorted({tuple(f.split("_locked_")[0].rsplit("_robot0_joint", 1)) for f in os.listdir(d50)
                        if f.endswith("_n50.json") and "_prio_" in f})
        nrows = []
        for t, j in conds:
            rec = {m: load(os.path.join(d50, f"{t}_robot0_joint{j}_locked_{m}_n50.json")) for m in N50_METHODS}
            if any(rec[m] is None for m in ["b1", "ik", "ik_pose", "prio"]):
                continue
            row = [f"{TASK_LABEL.get(t, t)} J{j}"]
            for m in N50_METHODS:
                row.append(f"{rec[m]['score']:.2f}" if rec[m] else "--")
            P = rec["prio"]
            for m in ["ik", "ik_pose"]:
                if P and rec[m] and P["pe"] and rec[m]["pe"]:
                    b, c = paired(P["pe"], rec[m]["pe"]); row.append(f"{b}:{c} ({fmt_p(mcnemar(b, c))})")
                else:
                    row.append("--")
            nrows.append(row)
        if nrows:
            write_table(OUT, "tab_n50", ["Condition"] + [N50_LABEL[m] for m in N50_METHODS] +
                        ["prio vs pos", "prio vs pose"], nrows,
                        "Fresh-seed replication (locked, $n{=}50$, seeds 10020--10069).", "tab:n50")
            report.append("tab_n50")

    # Table 5: ablation of the correction (locked, 28 conditions, paired vs Priority IK)
    ABL = [("prio", "Priority IK", "--", None),
           ("prio_rev", "Orientation-first priority", "task order reversed", "libero_fault_sweep_locked_prio_rev"),
           ("rg_prio", "RG-DDPM + Priority IK", "sampling-time guidance added", "libero_fault_sweep_locked_rg_prio"),
           ("rg", "RG-DDPM", "guidance + weighted IK (pose)", None),
           ("pose", "W-IK pose", "weighted IK, $w_r{=}1.0$", None),
           ("pos", "W-IK pos", "weighted IK, $w_r{=}0.05$", None)]
    JA = [1, 3, 5, 6, 7]
    def rec_of(m, d, t, j):
        return load(os.path.join(R, d, fname("locked", t, j))) if d else data.get(("locked", t, j, m))
    arows = []
    for m, lab, what, d in ABL:
        recs = {(t, j): rec_of(m, d, t, j) for t in TASKS for j in JOINTS}
        if any(r is None or r["score"] is None for r in recs.values()):
            print(f"  ablation {m}: incomplete -> skipped"); continue
        row = [lab, what] + [f"{np.mean([recs[(t, j)]['score'] for t in TASKS]):.2f}" for j in JA]
        row.append(f"{np.mean([r['score'] for r in recs.values()]):.2f}")
        if m == "prio":
            row.append("--")
        else:
            B = Cc = 0
            for (t, j), r in recs.items():
                P = data.get(("locked", t, j, "prio"))
                if P and P["pe"] and r["pe"]:
                    b, c = paired(P["pe"], r["pe"]); B += b; Cc += c
            row.append(f"{B}:{Cc} ({fmt_p(mcnemar(B, Cc))})")
        arows.append(row)
    if len(arows) >= 2:
        write_table(OUT, "tab_ablation",
                    ["Variant", "Change", "J1", "J3", "J5", "J6", "J7", "Mean (28)", "Prio:variant"], arows,
                    "Ablation of the correction on locked faults (mean success over 4 tasks per joint; J2/J4 are 0 for "
                    "every method). Paired = episodes only Priority IK solved : only the variant solved.",
                    "tab:ablation")
        report.append("tab_ablation")

    layer1(data, a.layer1, OUT, report)
    print("wrote:", ", ".join(report))
    print("output dir:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
