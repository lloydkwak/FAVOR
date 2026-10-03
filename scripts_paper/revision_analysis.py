"""
Revision analyses (reviewer requests), from results/ only. Missing inputs are skipped.

  tab_nofault          healthy (no-fault) success per task                   results/libero_nofault
  tab_wik_sweep        W-IK over w_r and rho vs Priority IK (locked J1,3,5,6,7 x 4 tasks)
  fig_wik_sweep        the same as curves: proximal (J1,J3) and distal (J6,J7)
  tab_ablation_ext     order/damping and prioritized-guidance variants on the same 20 conditions
  tab_stats_condition  condition-level tests: sign test, Wilcoxon signed-rank (Holm-adjusted),
                       cluster-bootstrap 95% CI of the mean difference, optional GLMM
  tab_main_ci          Table I means with cluster-bootstrap 95% CIs (resampling conditions)
  tab_latency          per-call latency                                       results/latency
  analysis/j4_range.md         J4 weakness across fault levels, with kinematic residuals
  analysis/fig3_outliers.md    conditions where the kinematic diagnosis and the outcome disagree

Usage (host, repo root): python scripts_paper/revision_analysis.py [--results results] [--out paper]
"""
import argparse, glob, json, os, re, sys
from math import comb

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_paper_figures as M                     # loaders, table writer, colours, rounding
import matplotlib.pyplot as plt

JR = [1, 3, 5, 6, 7]
LEVELS = M.LEVELS
SWEEP_DIR = "libero_fault_sweep_locked_{}"
EXISTING_WIK = {(0.05, 0.10): "libero_fault_sweep_locked_ik", (1.00, 0.10): "libero_fault_sweep_locked_ik_pose"}
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def score(rec):
    return None if rec is None else rec["score"]


def sign_test(d):
    d = [x for x in d if abs(x) > 1e-12]
    n, k = len(d), sum(1 for x in d if x > 0)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)


def wilcoxon(d):
    """Two-sided Wilcoxon signed-rank (zeros dropped, average ranks for ties, normal approx
    with tie correction; exact enumeration for n <= 12)."""
    d = np.asarray([x for x in d if abs(x) > 1e-12], float)
    n = len(d)
    if n == 0:
        return 1.0
    a = np.abs(d); o = np.argsort(a); r = np.empty(n); r[o] = np.arange(1, n + 1)
    for u in np.unique(a):
        m = a == u; r[m] = r[m].mean()
    w = r[d > 0].sum()
    if n <= 12:
        tot, cnt = 0, 0
        ranks = r
        lo = min(w, ranks.sum() - w)
        # exact distribution of W+ over sign flips
        from itertools import product
        for s in product((0, 1), repeat=n):
            tot += 1
            cnt += float(np.dot(s, ranks)) <= lo + 1e-9
        return min(1.0, 2 * cnt / tot)
    mu = n * (n + 1) / 4
    _, c = np.unique(a, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (c ** 3 - c).sum() / 48
    z = (abs(w - mu) - 0.5) / np.sqrt(var)
    from math import erfc, sqrt
    return erfc(z / sqrt(2))


def holm(ps):
    idx = np.argsort(ps); out = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); out[i] = run
    return out


def boot_ci(d, B=10000):
    d = np.asarray(d, float)
    if len(d) < 2:
        return (np.nan, np.nan)
    bs = d[RNG.integers(0, len(d), size=(B, len(d)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pooled(Pdict, Qdict):
    B = C = 0
    for k, P in Pdict.items():
        Q = Qdict.get(k)
        if P and Q and P["pe"] and Q["pe"]:
            b, c = M.paired(P["pe"], Q["pe"]); B += b; C += c
    return B, C


def fmt_ci(lo, hi):
    return f"[{lo:+.2f}, {hi:+.2f}]"


# ---------------------------------------------------------------- analyses
def nofault(R, OUT, rep):
    rows = []
    for t in M.TASKS:
        r = M.load(os.path.join(R, "libero_nofault", f"{t}.json"))
        if r and r["score"] is not None:
            rows.append([M.TASK_LABEL[t], M.r2(r["score"]), len(r["pe"] or {})])
    if rows:
        M.write_table(OUT, "tab_nofault", ["Task", "Healthy success (B1)", "n"], rows,
                      "Success without any fault (same seeds 10000--10019).", "tab:nofault")
        rep.append("tab_nofault")


def wik_sweep(R, data, OUT, rep):
    cfgs = dict(EXISTING_WIK)
    for d in glob.glob(os.path.join(R, SWEEP_DIR.format("wik_w*_r*"))):
        m = re.search(r"wik_w(\d{3})_r(\d{3})$", d)
        if m:
            cfgs[(int(m.group(1)) / 100, int(m.group(2)) / 100)] = os.path.basename(d)
    if len(cfgs) <= 2:
        print("  wik sweep: no new configs yet -> skipped"); return
    P = {(t, j): data.get(("locked", t, j, "prio")) for t in M.TASKS for j in JR}
    rows, curves = [], {}
    for (w, rho), d in sorted(cfgs.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        recs = {(t, j): M.load(os.path.join(R, d, M.fname("locked", t, j))) for t in M.TASKS for j in JR}
        if any(score(r) is None for r in recs.values()):
            print(f"  wik sweep w={w} rho={rho}: incomplete -> skipped"); continue
        perj = {j: np.mean([recs[(t, j)]["score"] for t in M.TASKS]) for j in JR}
        allm = np.mean([r["score"] for r in recs.values()])
        b, c = pooled(P, recs)
        dif = [P[k]["score"] - recs[k]["score"] for k in recs]
        rows.append([f"{w:.2f}", f"{rho:.2f}"] + [M.r2(perj[j]) for j in JR] + [M.r2(allm),
                     f"{b}:{c} ({M.fmt_p(M.mcnemar(b, c))})", f"{sum(x > 0 for x in dif)}/{sum(x < 0 for x in dif)}"])
        curves[(w, rho)] = dict(prox=np.mean([perj[1], perj[3]]), dist=np.mean([perj[6], perj[7]]), all=allm)
    pj = {j: np.mean([P[(t, j)]["score"] for t in M.TASKS]) for j in JR}
    rows.append(["\\multicolumn{2}{l}{Priority IK}"] + [M.r2(pj[j]) for j in JR] +
                [M.r2(np.mean([p["score"] for p in P.values()])), "--", "--"])
    M.write_table(OUT, "tab_wik_sweep", ["$w_r$", "$\\rho$", "J1", "J3", "J5", "J6", "J7", "Mean (20)",
                                         "Prio:W-IK", "cond. +/-"], rows,
                  "Weighted IK over the orientation weight $w_r$ and posture weight $\\rho$ on locked faults "
                  "(4 tasks $\\times$ J1, J3, J5, J6, J7). Paired = episodes only Priority IK solved : only W-IK solved; "
                  "cond. +/- = conditions where Priority IK was better / worse.", "tab:wik_sweep")
    rep.append("tab_wik_sweep")
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), sharey=True)
    for ax, key, title, pv in [(axs[0], "prox", "Proximal (J1, J3)", np.mean([pj[1], pj[3]])),
                               (axs[1], "dist", "Distal (J6, J7)", np.mean([pj[6], pj[7]]))]:
        for rho, ls, mk in [(0.10, "-", "o"), (0.0, "--", "s")]:
            pts = sorted((w, v[key]) for (w, r), v in curves.items() if abs(r - rho) < 1e-9)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker=mk, ms=3.5, lw=1.2,
                        color=M.COLOR["pos"] if rho > 0 else "#8A5A5D", label=f"W-IK, $\\rho$={rho:g}")
        ax.axhline(pv, color=M.COLOR["prio"], lw=1.6, label="Priority IK")
        ax.set_xscale("log"); ax.set_xlabel("orientation weight $w_r$ [m/rad]"); ax.set_title(title, pad=2)
        ax.set_ylim(0, 1); ax.yaxis.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
    axs[0].set_ylabel("Success rate (locked)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7)
    fig.tight_layout()
    M.savefig(fig, OUT, "fig_wik_sweep"); rep.append("fig_wik_sweep")


def ablation_ext(R, data, OUT, rep):
    V = [("prio", "Priority IK", None), ("prio_rev", "Orientation-first ($\\lambda$ on position 0.2)", "prio_rev"),
         ("prio_rev_l001", "Orientation-first ($\\lambda$ on position 0.01)", "prio_rev_l001"),
         ("rg_prio", "RG-DDPM (weighted internal) + Priority IK", "rg_prio"),
         ("rg_prioint", "RG-DDPM (prioritized internal) + Priority IK", "rg_prioint"),
         ("rg", "RG-DDPM (weighted internal) + W-IK pose", None)]
    P = {(t, j): data.get(("locked", t, j, "prio")) for t in M.TASKS for j in JR}
    rows = []
    for m, lab, d in V:
        recs = {(t, j): (M.load(os.path.join(R, SWEEP_DIR.format(d), M.fname("locked", t, j))) if d
                         else data.get(("locked", t, j, m))) for t in M.TASKS for j in JR}
        if any(score(r) is None for r in recs.values()):
            print(f"  ablation_ext {m}: incomplete -> skipped"); continue
        row = [lab] + [M.r2(np.mean([recs[(t, j)]["score"] for t in M.TASKS])) for j in JR]
        row.append(M.r2(np.mean([r["score"] for r in recs.values()])))
        if m == "prio":
            row.append("--")
        else:
            b, c = pooled(P, recs); row.append(f"{b}:{c} ({M.fmt_p(M.mcnemar(b, c))})")
        rows.append(row)
    if len(rows) >= 2 and any("l001" in r[0] or "prioritized" in r[0] for r in rows):
        M.write_table(OUT, "tab_ablation_ext", ["Variant", "J1", "J3", "J5", "J6", "J7", "Mean (20)", "Prio:variant"],
                      rows, "Order, damping and prioritized-guidance variants on locked J1, J3, J5, J6, J7 "
                      "(4 tasks).", "tab:ablation_ext")
        rep.append("tab_ablation_ext")


def condition_stats(data, OUT, rep, glmm):
    comps = [("b1", "B1"), ("eci", "E-C-I"), ("rg", "RG-DDPM"), ("pos", "W-IK pos"), ("pose", "W-IK pose"),
             ("best", "Best W-IK")]
    rows, pvals_s, pvals_w = [], [], []
    for level in LEVELS + ["all"]:
        lv = LEVELS if level == "all" else [level]
        for m, lab in comps:
            d = []
            for L in lv:
                for t in M.TASKS:
                    for j in M.JOINTS:
                        P = data.get((L, t, j, "prio"))
                        Q = M.best_of(data, L, t, j) if m == "best" else data.get((L, t, j, m))
                        if P and Q:
                            d.append(P["score"] - Q["score"])
            if len(d) < 5:
                continue
            ps, pw = sign_test(d), wilcoxon(d)
            lo, hi = boot_ci(d)
            rows.append([level, lab, len(d), f"{np.mean(d):+.3f}", fmt_ci(lo, hi),
                         f"{sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}", ps, pw])
            pvals_s.append(ps); pvals_w.append(pw)
    if not rows:
        return
    hs, hw = holm(np.array(pvals_s)), holm(np.array(pvals_w))
    out = [r[:6] + [M.fmt_p(hs[i]), M.fmt_p(hw[i])] for i, r in enumerate(rows)]
    M.write_table(OUT, "tab_stats_condition",
                  ["Level", "vs", "conds", "mean diff", "95\\% CI", "better/worse", "sign (Holm)", "Wilcoxon (Holm)"],
                  out, "Condition-level comparison of Priority IK with each method: mean difference in success "
                  "rate over conditions, cluster-bootstrap 95\\% CI (10{,}000 resamples of conditions), number of "
                  "conditions where Priority IK was better/worse, and sign and Wilcoxon signed-rank tests over "
                  "conditions with Holm correction across all rows.", "tab:stats_condition")
    rep.append("tab_stats_condition")
    if glmm:
        glmm_fit(data, OUT, rep)


def glmm_fit(data, OUT, rep):
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as e:
        print(f"  GLMM skipped ({e}); pip install statsmodels pandas"); return
    rows = []
    for level in LEVELS:
        for m in ["b1", "eci", "rg", "pos", "pose"]:
            recs = []
            for t in M.TASKS:
                for j in M.JOINTS:
                    P, Q = data.get((level, t, j, "prio")), data.get((level, t, j, m))
                    if not (P and Q and P["pe"] and Q["pe"]):
                        continue
                    for s, v in P["pe"].items():
                        recs.append(dict(y=int(v > .5), prio=1, cond=f"{t}{j}", ep=f"{t}{j}_{s}"))
                    for s, v in Q["pe"].items():
                        recs.append(dict(y=int(v > .5), prio=0, cond=f"{t}{j}", ep=f"{t}{j}_{s}"))
            if len(recs) < 200:
                continue
            df = pd.DataFrame(recs)
            md = BinomialBayesMixedGLM.from_formula("y ~ prio", {"cond": "0 + C(cond)", "ep": "0 + C(ep)"}, df)
            fit = md.fit_vb()
            k = list(md.exog_names).index("prio")
            rows.append([level, m, f"{fit.fe_mean[k]:+.2f}", f"{fit.fe_sd[k]:.2f}",
                         f"{np.exp(fit.fe_mean[k]):.1f}"])
            print(f"  GLMM {level} vs {m}: log-odds {fit.fe_mean[k]:+.2f} (sd {fit.fe_sd[k]:.2f})")
    if rows:
        M.write_table(OUT, "tab_glmm", ["Level", "vs", "log-odds (Prio)", "posterior sd", "odds ratio"], rows,
                      "Mixed-effects logistic regression (variational Bayes): success ~ method, random intercepts "
                      "for condition and for the shared episode (seed).", "tab:glmm")
        rep.append("tab_glmm")


def main_ci(data, OUT, rep):
    rows = []
    ms = ["b1", "eci", "rg", "pos", "pose", "prio"]
    for level in LEVELS:
        row = [level]
        for m in ms:
            v = [data[(level, t, j, m)]["score"] for t in M.TASKS for j in M.JOINTS if (level, t, j, m) in data]
            if len(v) < 5:
                row.append("--"); continue
            lo, hi = boot_ci(v)
            row.append(f"{M.r2(np.mean(v))} [{lo:.2f}, {hi:.2f}]")
        rows.append(row)
    M.write_table(OUT, "tab_main_ci", ["Level", "B1", "E-C-I", "RG-DDPM", "W-IK pos", "W-IK pose", "Priority IK"],
                  rows, "Mean success with cluster-bootstrap 95\\% CIs over conditions.", "tab:main_ci")
    rep.append("tab_main_ci")


def latency(R, OUT, rep):
    fs = sorted(glob.glob(os.path.join(R, "latency", "*.json")))
    if not fs:
        return
    js = json.load(open(fs[0]))
    lab = {"b1": "B1", "pos": "W-IK pos", "pose": "W-IK pose", "prio": "Priority IK", "rg": "RG-DDPM"}
    rows = [[lab[m], f"{v['mean_ms']:.0f}", f"{v['median_ms']:.0f}", f"{v['p95_ms']:.0f}", f"{v['overhead_vs_b1_ms']:+.0f}"]
            for m, v in js["methods"].items()]
    M.write_table(OUT, "tab_latency", ["Method", "mean [ms]", "median", "p95", "vs B1"], rows,
                  f"Wall-clock time per policy call (one 16-step chunk for 5 parallel environments, 8 steps "
                  f"executed), {M.TASK_LABEL.get(js['task'], js['task'])} J{js['joint']} locked, {js['gpu']}.",
                  "tab:latency")
    rep.append("tab_latency")


def j4_report(R, data, layer1, OUT, rep):
    L1 = {}
    if os.path.exists(layer1):
        import csv
        for r in csv.DictReader(open(layer1)):
            L1[(r["level"], r["task"], int(r["joint"][-1]))] = r
    lines = ["# J4 across fault levels", "",
             "Success per task (B1 / W-IK pos / W-IK pose / Priority IK) and the policy-free kinematic errors "
             "(override = fault simply overrides the joint; residual = left after W-IK retargeting, mm).", ""]
    for level in LEVELS:
        lines += [f"## {level}", "", "| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | "
                  "Prio dq_free | W-IK pos dq_free |", "|---|---|---|---|---|---|---|---|---|"]
        for t in M.TASKS:
            g = lambda m: data.get((level, t, 4, m))
            v = [g(m) for m in ("b1", "pos", "pose", "prio")]
            l1 = L1.get((level, t, 4), {})
            ps = (g("prio") or {}).get("raw", {}).get("prio_stats") or {}
            ks = (g("pos") or {}).get("raw", {}).get("ik_stats") or {}
            dqp = next((ps[k] for k in ps if "dq" in k), None)
            dqw = ks.get("dq_free_max_mean")
            lines.append(f"| {M.TASK_LABEL[t]} | " + " | ".join("–" if x is None else f"{x['score']:.2f}" for x in v) +
                         f" | {l1.get('override_pos_mm', '–')} | {l1.get('ik_pos_mm', '–')} | "
                         f"{'–' if dqp is None else f'{dqp:.3f}'} | {'–' if dqw is None else f'{dqw:.3f}'} |")
        lines.append("")
    lines += ["Reading guide: if Priority IK moves the free joints much more than W-IK pos (dq_free) while the "
              "residual is small, the drop is consistent with the missing posture term (W-IK keeps the policy's "
              "posture via rho); if the residual is large, the fault is only partly reachable and the two methods "
              "fail in different ways.", ""]
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    open(os.path.join(OUT, "analysis", "j4_range.md"), "w").write("\n".join(lines))
    rep.append("analysis/j4_range.md")


def fig3_outliers(R, data, layer1, OUT, rep):
    if not os.path.exists(layer1):
        return
    import csv
    nf = {t: score(M.load(os.path.join(R, "libero_nofault", f"{t}.json"))) for t in M.TASKS}
    rows = list(csv.DictReader(open(layer1)))
    a, b = [], []
    for r in rows:
        level, t, j = r["level"], r["task"], int(r["joint"][-1])
        B, P = data.get((level, t, j, "b1")), data.get((level, t, j, "prio"))
        ov, res = float(r["override_pos_mm"]), float(r["ik_pos_mm"])
        if B and ov < 0.1:
            a.append((B["score"], level, t, j, ov, nf.get(t)))
        if P and res < 3.0 and P["score"] < 0.3:
            b.append((P["score"], level, t, j, res, nf.get(t)))
    f = lambda x: "–" if x is None else f"{x:.2f}"
    lines = ["# Fig. 3 outliers", "",
             "## Left panel: override error ~ 0 (< 0.1 mm), B1 success", "",
             "Near-zero kinematic error means the demonstrations barely move the faulty joint, so the fault "
             "costs nothing kinematically; the remaining spread should then be the policy's own success rate "
             "(compare with the healthy rate).", "",
             "| B1 | level | task | joint | override mm | healthy |", "|---|---|---|---|---|---|"]
    lines += [f"| {s:.2f} | {lv} | {M.TASK_LABEL[t]} | J{j} | {ov:.3f} | {f(h)} |" for s, lv, t, j, ov, h in sorted(a)]
    lines += ["", "## Right panel: residual < 3 mm but Priority IK success < 0.3", "",
              "| Priority IK | level | task | joint | residual mm | healthy |", "|---|---|---|---|---|---|"]
    lines += [f"| {s:.2f} | {lv} | {M.TASK_LABEL[t]} | J{j} | {re_:.2f} | {f(h)} |" for s, lv, t, j, re_, h in sorted(b)]
    lines += ["", "The residual is computed on demonstration waypoints with W-IK; a small residual there does "
              "not guarantee that the policy's own (shifted) trajectories stay reachable, and Priority IK's "
              "residual differs from W-IK's.", ""]
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    open(os.path.join(OUT, "analysis", "fig3_outliers.md"), "w").write("\n".join(lines))
    rep.append("analysis/fig3_outliers.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--layer1", default="analysis_out/layer1_v2.csv")
    ap.add_argument("--glmm", action="store_true", help="also fit the mixed-effects model (needs statsmodels)")
    a = ap.parse_args()
    data = M.collect(a.results)
    rep = []
    nofault(a.results, a.out, rep)
    wik_sweep(a.results, data, a.out, rep)
    ablation_ext(a.results, data, a.out, rep)
    condition_stats(data, a.out, rep, a.glmm)
    main_ci(data, a.out, rep)
    latency(a.results, a.out, rep)
    j4_report(a.results, data, a.layer1, a.out, rep)
    fig3_outliers(a.results, data, a.layer1, a.out, rep)
    print("wrote:", ", ".join(rep))


if __name__ == "__main__":
    main()
