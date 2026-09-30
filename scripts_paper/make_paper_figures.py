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
    "pos":      ("B-IK pos",             "libero_fault_sweep_locked_ik",       "libero_fault_sweep_range_ik"),
    "pose":     ("B-IK pose",            "libero_fault_sweep_locked_ik_pose",  "libero_fault_sweep_range_ik_pose"),
    "prio":     ("Priority IK (ours)",   "libero_fault_sweep_locked_prio",     "libero_fault_sweep_range_prio"),
}
SHORT = {"b1": "B1", "random_n": "Rand-N", "select": "Select", "eci": "E-C-I", "rg": "RG-DDPM",
         "pos": "B-IK pos", "pose": "B-IK pose", "prio": "\\textbf{Prio-IK}"}
COLOR = {"b1": "#9E9C9C", "random_n": "#CFCFCF", "select": "#B8C4D9", "eci": "#163A78", "rg": "#5B7DB8",
         "pos": "#E39A9E", "pose": "#F2C4C6", "prio": "#B4131C", "best": "#5A5A5A"}
MS = {  # robustness to misspecified fault knowledge (Priority IK)
    "rs050":  ("window $\\times$0.5",  "range", "moderate", "libero_fault_sweep_range_ms_rs050"),
    "rs150":  ("window $\\times$1.5",  "range", "moderate", "libero_fault_sweep_range_ms_rs150"),
    "lop010": ("lock +0.1 rad", "locked", "locked", "libero_fault_sweep_locked_ms_lop010"),
    "lom010": ("lock $-$0.1 rad", "locked", "locked", "libero_fault_sweep_locked_ms_lom010"),
}
N50_METHODS = ["b1", "eci", "ik", "ik_pose", "prio"]
N50_LABEL = {"b1": "B1", "eci": "E-C-I", "ik": "B-IK pos", "ik_pose": "B-IK pose", "prio": "Priority IK"}

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
        fig.savefig(os.path.join(out, "figs", f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper")
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
    header = ["Fault level"] + [SHORT[m] for m in cols] + ["Oracle$^\\dagger$"]
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
                "per fault level. $^\\dagger$Oracle that picks, per condition, the better B-IK setting after seeing the "
                "results. Superscripts give the number of conditions when fewer than 28 were run.", "tab:main")
    report.append("tab_main")

    # Table 2: paired tests Priority IK vs others
    if any(k[3] == "prio" for k in data):
        comp = [m for m in ["b1", "eci", "rg", "pos", "pose"] if m in have] + ["best"]
        header = ["Fault level"] + [("vs " + (SHORT[m] if m != "best" else "Oracle$^\\dagger$")) for m in comp]
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
            im = ax.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
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
                label="best(pos,pose) oracle")
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
        if n:
            rrows.append([lab, level, n, f"{np.mean(ex_s):.2f}", f"{np.mean(ms_s):.2f}",
                          f"{np.mean(ms_s) - np.mean(ex_s):+.2f}", f"{b}:{c} ({fmt_p(mcnemar(b, c))})"])
            bars.append((f"{lab}\n({n} cond.)", np.mean(ex_s), np.mean(ms_s)))
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
        ax.legend(frameon=False, ncol=2); ax.yaxis.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
        savefig(fig, OUT, "fig_misspec"); report.append("tab_misspec, fig_misspec")

    # Table 4: n=50 fresh-seed replication
    d50 = os.path.join(R, "libero_confirm_n50_fresh")
    if os.path.isdir(d50):
        conds = sorted({tuple(f.split("_locked_")[0].rsplit("_robot0_joint", 1)) for f in os.listdir(d50)
                        if f.endswith("_n50.json") and "_prio_" in f})
        nrows = []
        for t, j in conds:
            rec = {m: load(os.path.join(d50, f"{t}_robot0_joint{j}_locked_{m}_n50.json")) for m in N50_METHODS}
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

    print("wrote:", ", ".join(report))
    print("output dir:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
