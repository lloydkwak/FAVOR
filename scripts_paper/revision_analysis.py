"""
Additional analyses, from results/ only. Missing inputs are skipped.

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
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def nofault(R, OUT, rep):
    rows = []
    for t in M.TASKS:
        r20 = M.load(os.path.join(R, "libero_nofault", f"{t}.json"))
        r100 = M.load(os.path.join(R, "libero_nofault100", f"{t}.json"))
        if not (r20 or r100):
            continue
        row = [M.TASK_LABEL[t], M.r2(r20["score"]) if r20 else "--"]
        if r100 and r100["pe"]:
            k, n = sum(v > .5 for v in r100["pe"].values()), len(r100["pe"])
            lo, hi = wilson(k, n)
            row += [M.r2(k / n), f"[{lo:.2f}, {hi:.2f}]"]
        else:
            row += ["--", "--"]
        rows.append(row)
    if rows:
        M.write_table(OUT, "tab_nofault", ["Task", "Healthy, seeds 10000--10019", "Healthy, $n{=}100$", "95\\% CI (Wilson)"],
                      rows, "Success without any fault. The $n{=}20$ column uses the evaluation seeds; the $n{=}100$ "
                      "column (seeds 10000--10099) gives the reference rate with its Wilson interval.", "tab:nofault")
        rep.append("tab_nofault")


WIK_RE = r"wik_w(\d{3,4})_r(\d{3})(?:_dm(\d))?(?:_it(\d+))?$"


def _wik_dirs(R):
    """(w_r, rho, damping exponent or None, iterations or None) -> result dir."""
    cfgs = {(w, r, None, None): d for (w, r), d in EXISTING_WIK.items()}
    for d in glob.glob(os.path.join(R, SWEEP_DIR.format("wik_w*_r*"))):
        m = re.search(WIK_RE, d)
        if m:
            w = int(m.group(1)) / (100 if len(m.group(1)) == 3 else 1000)
            cfgs[(w, int(m.group(2)) / 100, int(m.group(3)) if m.group(3) else None,
                  int(m.group(4)) if m.group(4) else None)] = os.path.basename(d)
    return cfgs


def wik_sweep(R, data, OUT, rep):
    cfgs = _wik_dirs(R)
    if len(cfgs) <= 2:
        print("  wik sweep: no new configs yet -> skipped"); return
    P = {(t, j): data.get(("locked", t, j, "prio")) for t in M.TASKS for j in JR}
    rows, curves = [], {}
    for (w, rho, dm, it), d in sorted(cfgs.items(), key=lambda kv: (kv[0][2] is not None, kv[0][3] or 30,
                                                                    -kv[0][1], kv[0][0])):
        recs = {(t, j): M.load(os.path.join(R, d, M.fname("locked", t, j))) for t in M.TASKS for j in JR}
        if any(score(r) is None for r in recs.values()):
            print(f"  wik sweep w={w} rho={rho} dm={dm} it={it}: incomplete -> skipped"); continue
        perj = {j: np.mean([recs[(t, j)]["score"] for t in M.TASKS]) for j in JR}
        allm = np.mean([r["score"] for r in recs.values()])
        b, c = pooled(P, recs)
        dif = [P[k]["score"] - recs[k]["score"] for k in recs]
        damp = "$10^{-6}$" if dm is None else f"$10^{{-{dm}}}$"
        rows.append([f"{w:g}", f"{rho:.2f}", damp, str(it or 30)] + [M.r2(perj[j]) for j in JR] + [M.r2(allm),
                     f"{b}:{c} ({M.fmt_p(M.mcnemar(b, c))})", f"{sum(x > 0 for x in dif)}/{sum(x < 0 for x in dif)}"])
        if dm is None and it is None:
            curves[(w, rho)] = dict(prox=np.mean([perj[1], perj[3]]), dist=np.mean([perj[6], perj[7]]), all=allm)
    pj = {j: np.mean([P[(t, j)]["score"] for t in M.TASKS]) for j in JR}
    rows.append(["\\multicolumn{4}{l}{Priority IK ($\\lambda_1^2{=}10^{-4}$, 30 it.)}"] + [M.r2(pj[j]) for j in JR] +
                [M.r2(np.mean([p["score"] for p in P.values()])), "--", "--"])
    M.write_table(OUT, "tab_wik_sweep", ["$w_r$", "$\\rho$", "damping", "iter.", "J1", "J3", "J5", "J6", "J7", "Mean (20)",
                                         "Prio:W-IK", "cond. +/-"], rows,
                  "Weighted IK over the orientation weight $w_r$, posture weight $\\rho$ and Levenberg--Marquardt "
                  "damping on locked faults, 4 tasks $\\times$ J1, J3, J5, J6, J7. J2 and J4 are excluded because every "
                  "method is near zero on them (Table III), so they cannot separate the settings. Paired = episodes only "
                  "Priority IK solved : only W-IK solved; cond. +/- = conditions where Priority IK was better / worse.",
                  "tab:wik_sweep")
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


def _variant_recs(R, data, level, m, d):
    kind = "locked" if level == "locked" else "range"
    return {(t, j): (M.load(os.path.join(R, f"libero_fault_sweep_{kind}_{d}", M.fname(level, t, j))) if d
                     else data.get((level, t, j, m))) for t in M.TASKS for j in JR}


def _variant_rows(R, data, level, V, partial=False):
    """partial=True keeps a variant with at least half of the 20 conditions; its row is computed on the
    conditions it has (marked with a dagger and the count), so its mean is not comparable to the others."""
    P = {(t, j): data.get((level, t, j, "prio")) for t in M.TASKS for j in JR}
    rows = []
    for m, lab, d in V:
        recs = _variant_recs(R, data, level, m, d)
        have = {k: r for k, r in recs.items() if score(r) is not None}
        if len(have) < len(recs):
            if not (partial and len(have) >= len(recs) // 2):
                print(f"  {level} {m}: incomplete ({len(have)}/{len(recs)}) -> skipped"); continue
            print(f"  {level} {m}: partial ({len(have)}/{len(recs)}) -> kept, marked")
            lab = f"{lab}$^\\dagger$ ({len(have)}/{len(recs)})"
            recs = have
        row = [lab] + [M.r2(np.mean(v)) if (v := [recs[(t, j)]["score"] for t in M.TASKS if (t, j) in recs]) else "--"
                       for j in JR]
        row.append(M.r2(np.mean([r["score"] for r in recs.values()])))
        if m == "prio":
            row.append("--")
        else:
            b, c = pooled(P, recs); row.append(f"{b}:{c} ({M.fmt_p(M.mcnemar(b, c))})")
        rows.append(row)
    return rows


def ablation_ext(R, data, OUT, rep):
    V = [("prio", "Priority IK", None),
         ("prio_rev", "Orientation first ($\\lambda$ on position 0.2)", "prio_rev"),
         ("prio_rev_l001", "Orientation first ($\\lambda$ on position 0.01)", "prio_rev_l001"),
         ("rg", "RG: weighted + posture internal, W-IK pose exec.", None),
         ("rg_prio", "RG: weighted + posture internal", "rg_prio"),
         ("rg_wint_nob", "RG: weighted internal, no posture term, no budget", "rg_wint_nob"),
         ("rg_prioint_b03", "RG: prioritized internal, budget 0.3 rad", "rg_prioint_b03"),
         ("rg_prioint", "RG: prioritized internal, no budget", "rg_prioint")]
    rows = _variant_rows(R, data, "locked", V, partial=True)
    if len(rows) >= 2:
        M.write_table(OUT, "tab_ablation_ext", ["Variant", "J1", "J3", "J5", "J6", "J7", "Mean (20)", "Prio:variant"],
                      rows, "Order, damping and denoising-time guidance variants on locked J1, J3, J5, J6, J7 "
                      "(4 tasks; J2/J4 excluded as in Table~\\ref{tab:wik_sweep}). Unless stated otherwise, RG variants "
                      "execute with Priority IK; budget = maximum motion of each healthy joint in the internal "
                      "correction. $^\\dagger$Incomplete variant: computed on the conditions it has (count given); see "
                      "Table~\\ref{tab:rg_factors} for comparisons on a common condition set.", "tab:ablation_ext")
        rep.append("tab_ablation_ext")
    rg_factors(R, data, OUT, rep)
    Vm = [("prio", "Priority IK", None), ("b1", "B1", None), ("pos", "W-IK pos", None), ("pose", "W-IK pose", None),
          ("rg", "RG: weighted internal, W-IK pose exec.", None),
          ("rg_prioint", "RG: prioritized internal, Priority IK exec.", "rg_prioint")]
    rows = _variant_rows(R, data, "moderate", Vm)
    if any("prioritized" in r[0] for r in rows):
        M.write_table(OUT, "tab_rg_moderate", ["Method", "J1", "J3", "J5", "J6", "J7", "Mean (20)", "Prio:method"],
                      rows, "Prioritized denoising-time guidance on moderate range faults (4 tasks $\\times$ J1, J3, "
                      "J5, J6, J7).", "tab:rg_moderate")
        rep.append("tab_rg_moderate")


def rg_factors(R, data, OUT, rep):
    """Which part of the RG internal correction matters, each factor isolated on the conditions that
    every compared variant has: motion budget, prioritized vs weighted, posture term."""
    V = {"prio": ("Priority IK", None), "rg_prio": ("RG: weighted + posture", "rg_prio"),
         "rg_wint_nob": ("RG: weighted, no posture, no budget", "rg_wint_nob"),
         "rg_prioint_b03": ("RG: prioritized, budget 0.3 rad", "rg_prioint_b03"),
         "rg_prioint": ("RG: prioritized, no budget", "rg_prioint")}
    recs = {m: _variant_recs(R, data, "locked", m, d) for m, (_, d) in V.items()}
    C = [k for k in recs["prio"] if all(score(recs[m][k]) is not None for m in V)]
    if len(C) < 5:
        print(f"  rg_factors: only {len(C)} common conditions -> skipped"); return
    tasks = sorted({t for t, _ in C}, key=M.TASKS.index)
    rows = []
    for m, (lab, _) in V.items():
        s = {k: recs[m][k]["score"] for k in C}
        pj = [M.r2(np.mean(v)) if (v := [s[k] for k in C if k[1] == j]) else "--" for j in JR]
        rows.append([lab] + pj + [M.r2(np.mean(list(s.values()))),
                                  M.r2(np.mean([v for k, v in s.items() if k[1] in (1, 3)])),
                                  M.r2(np.mean([v for k, v in s.items() if k[1] in (5, 6, 7)]))])
    M.write_table(OUT, "tab_rg_factors_means", ["Variant", "J1", "J3", "J5", "J6", "J7", "Mean", "Prox.", "Dist."],
                  rows, f"RG internal-correction variants on the {len(C)} locked conditions all of them have "
                  f"({', '.join(M.TASK_LABEL.get(t, t) for t in tasks)}; J1, J3, J5, J6, J7). "
                  "All execute with Priority IK.", "tab:rg_factors_means")
    comps = [("Motion budget", "rg_prioint", "rg_prioint_b03", "no budget vs 0.3 rad (both prioritized)"),
             ("Prioritized vs weighted", "rg_prioint", "rg_wint_nob", "both without budget or posture term"),
             ("Posture term", "rg_wint_nob", "rg_prio", "without vs with posture term$^\\ddagger$")]
    rows = []
    for name, a, b, note in comps:
        B = Cc = 0
        for k in C:
            x, y = M.paired(recs[a][k]["pe"], recs[b][k]["pe"]); B += x; Cc += y
        rows.append([name, note, f"{B}:{Cc}", M.fmt_p(M.mcnemar(B, Cc))])
    M.write_table(OUT, "tab_rg_factors", ["Factor", "Comparison", "Episodes A:B", "McNemar $p$"], rows,
                  f"One factor of the RG internal correction changed at a time, pooled over the {len(C)} common "
                  "conditions (A = first variant succeeds alone, B = second). $^\\ddagger$The two variants also differ "
                  "in iteration count (10 vs 5) and damping ($10^{-4}$ vs $10^{-6}$).", "tab:rg_factors", "llcc")
    rep += ["tab_rg_factors_means", "tab_rg_factors"]


def condition_stats(data, OUT, rep, glmm):
    """Primary family (pre-specified): Priority IK vs each method over all levels -> Holm within it.
    Per-level rows are exploratory: raw p-values, no correction."""
    comps = [("b1", "B1"), ("eci", "E-C-I"), ("rg", "RG-DDPM"), ("pos", "W-IK pos"), ("pose", "W-IK pose"),
             ("best", "Best W-IK")]
    rows = []
    for level in ["all"] + LEVELS:
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
            lo, hi = boot_ci(d)
            rows.append(dict(level=level, lab=lab, n=len(d), mean=np.mean(d), ci=fmt_ci(lo, hi),
                             bw=f"{sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}", ps=sign_test(d), pw=wilcoxon(d)))
    if not rows:
        return
    prim = [r for r in rows if r["level"] == "all"]
    hs, hw = holm(np.array([r["ps"] for r in prim])), holm(np.array([r["pw"] for r in prim]))
    for r, a_, b_ in zip(prim, hs, hw):
        r["hs"], r["hw"] = a_, b_
    out = []
    for r in rows:
        tag = "primary" if r["level"] == "all" else "exploratory"
        out.append([r["level"], r["lab"], tag, r["n"], f"{r['mean']:+.3f}", r["ci"], r["bw"],
                    M.fmt_p(r["ps"]), M.fmt_p(r["pw"]),
                    M.fmt_p(r["hs"]) if "hs" in r else "--", M.fmt_p(r["hw"]) if "hw" in r else "--"])
    M.write_table(OUT, "tab_stats_condition",
                  ["Level", "vs", "family", "conds", "mean diff", "95\\% CI", "better/worse", "sign p", "Wilcoxon p",
                   "sign (Holm)", "Wilcoxon (Holm)"],
                  out, "Condition-level comparison of Priority IK with each method: mean difference in success rate "
                  "over conditions, cluster-bootstrap 95\\% CI (10{,}000 resamples of conditions), conditions where "
                  "Priority IK was better/worse, and sign and Wilcoxon signed-rank tests over conditions. The primary "
                  "family (all 112 conditions, one test per compared method) is Holm-corrected; per-level rows are "
                  "exploratory and uncorrected.", "tab:stats_condition")
    rep.append("tab_stats_condition")
    if glmm:
        glmm_fit(data, OUT, rep)


def glmm_fit(data, OUT, rep):
    """success ~ method, with random intercepts for condition and for the shared episode, and a random
    slope of the method effect over conditions (variational Bayes)."""
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as e:
        print(f"  GLMM skipped ({e}); pip install statsmodels pandas"); return
    rows = []
    for level in LEVELS:                      # per level (a pooled fit over 112 conditions is too slow for VB)
        lv = [level]
        for m in ["b1", "eci", "rg", "pos", "pose"]:
            recs = []
            for L in lv:
                for t in M.TASKS:
                    for j in M.JOINTS:
                        P, Q = data.get((L, t, j, "prio")), data.get((L, t, j, m))
                        if not (P and Q and P["pe"] and Q["pe"]):
                            continue
                        c = f"{L}{t}{j}"
                        for s, v in P["pe"].items():
                            recs.append(dict(y=int(v > .5), prio=1, cond=c, ep=f"{c}_{s}"))
                        for s, v in Q["pe"].items():
                            recs.append(dict(y=int(v > .5), prio=0, cond=c, ep=f"{c}_{s}"))
            if len(recs) < 200:
                continue
            df = pd.DataFrame(recs)
            md = BinomialBayesMixedGLM.from_formula(
                "y ~ prio", {"cond": "0 + C(cond)", "slope": "0 + C(cond):prio", "ep": "0 + C(ep)"}, df)
            fit = md.fit_vb()
            k = list(md.exog_names).index("prio")
            names = list(md.vcp_names)
            sd_slope = float(np.exp(fit.vcp_mean[names.index("slope")])) if "slope" in names else np.nan
            mu, sd = fit.fe_mean[k], fit.fe_sd[k]
            rows.append([level, m, f"{mu:+.2f}", f"{sd:.2f}", f"[{mu - 1.96 * sd:+.2f}, {mu + 1.96 * sd:+.2f}]",
                         f"{np.exp(mu):.1f}", f"{sd_slope:.2f}"])
            print(f"  GLMM {level} vs {m}: log-odds {mu:+.2f} (sd {sd:.2f}), slope sd {sd_slope:.2f}")
    if rows:
        M.write_table(OUT, "tab_glmm", ["Level", "vs", "log-odds (Prio)", "posterior sd", "95\\% interval",
                                        "odds ratio", "sd of slope over cond."], rows,
                      "Mixed-effects logistic regression (variational Bayes): success $\\sim$ method, with random "
                      "intercepts for condition and for the shared episode (seed) and a random slope of the method "
                      "effect over conditions.", "tab:glmm")
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
                  f"executed; the IK corrects up to 8 waypoints $\\times$ 5 environments per call), "
                  f"{M.TASK_LABEL.get(js['task'], js['task'])} J{js['joint']} locked, {js['gpu']}. The base policy "
                  f"uses 100 DDPM steps and is itself slower than the 0.4~s that 8 control steps take; a DDIM sampler "
                  f"with fewer steps would reduce all rows alike.",
                  "tab:latency")
    rep.append("tab_latency")


def _mm(x):
    try:
        return f"{float(x):.1f}"
    except (TypeError, ValueError):
        return "–"


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
                         f" | {_mm(l1.get('override_pos_mm'))} | {_mm(l1.get('ik_pos_mm'))} | "
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
             "Every point with zero override error is J7. J7 rotates about the flange axis and everything "
             "after it lies on that axis, so locking J7 leaves the end-effector POSITION unchanged and only turns "
             "the gripper about its axis (yaw). The position-only diagnosis therefore reports 0 mm by construction, "
             "while the orientation error is what makes B1 fail at the locked and severe levels. A complete "
             "predictor needs an orientation term (e.g. the error of a fingertip point offset from the flange), or "
             "J7 has to be marked and discussed separately (Fig. 3 marks it).", "",
             "| B1 | level | task | joint | override mm | healthy |", "|---|---|---|---|---|---|"]
    lines += [f"| {s:.2f} | {lv} | {M.TASK_LABEL[t]} | J{j} | {ov:.3f} | {f(h)} |" for s, lv, t, j, ov, h in sorted(a)]
    lines += ["", "## Right panel: residual < 3 mm but Priority IK success < 0.3", "",
              "| Priority IK | level | task | joint | residual mm | healthy |", "|---|---|---|---|---|---|"]
    lines += [f"| {s:.2f} | {lv} | {M.TASK_LABEL[t]} | J{j} | {re_:.2f} | {f(h)} |" for s, lv, t, j, re_, h in sorted(b)]
    lines += ["", "Same cause: the residual is a position residual, and for J7 the remaining error is the gripper "
              "yaw. It is also computed on demonstration waypoints with W-IK, not on the policy's own trajectories "
              "with Priority IK.", ""]
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    open(os.path.join(OUT, "analysis", "fig3_outliers.md"), "w").write("\n".join(lines))
    rep.append("analysis/fig3_outliers.md")


# ---------------------------------------------------------------- round 3
def _sweep_row(P, recs, lab):
    perj = [M.r2(np.mean(v)) if (v := [recs[k]["score"] for k in recs if k[1] == j]) else "--" for j in JR]
    b, c = pooled({k: P[k] for k in recs}, recs)
    dif = [P[k]["score"] - recs[k]["score"] for k in recs if P.get(k)]
    return [lab] + perj + [M.r2(np.mean([r["score"] for r in recs.values()])),
                           f"{b}:{c} ({M.fmt_p(M.mcnemar(b, c))})",
                           f"{sum(x > 0 for x in dif)}/{sum(x < 0 for x in dif)}"]


def prio_lam2(R, data, OUT, rep):
    """Priority IK's secondary damping lambda_2 (main runs: 0.2) on locked J1, J3, J5, J6, J7."""
    P = {(t, j): data.get(("locked", t, j, "prio")) for t in M.TASKS for j in JR}
    found = []
    for d in glob.glob(os.path.join(R, SWEEP_DIR.format("prio_l2_*"))):
        m = re.search(r"prio_l2_(\d{3})$", d)
        if m:
            found.append((int(m.group(1)) / 100, os.path.basename(d)))
    if not found:
        return
    rows = []
    for lam, d in sorted(found + [(0.2, None)]):
        if d is None:
            recs = {k: v for k, v in P.items() if v}
            rows.append([f"{lam:g} (main)"] + _sweep_row(P, recs, "")[1:-2] + ["--", "--"]); continue
        recs = {(t, j): M.load(os.path.join(R, d, M.fname("locked", t, j))) for t in M.TASKS for j in JR}
        recs = {k: v for k, v in recs.items() if score(v) is not None}
        if len(recs) < len(JR) * len(M.TASKS) // 2:
            print(f"  prio lam2 {lam}: {len(recs)} conditions -> skipped"); continue
        lab = f"{lam:g}" + ("" if len(recs) == 20 else f"$^\\dagger$ ({len(recs)}/20)")
        rows.append(_sweep_row(P, recs, lab))
    M.write_table(OUT, "tab_prio_lam2", ["$\\lambda_2$", "J1", "J3", "J5", "J6", "J7", "Mean", "0.2:variant",
                                         "cond. +/-"], rows,
                  "Priority IK over its secondary (orientation) damping $\\lambda_2$ on locked faults, 4 tasks $\\times$ "
                  "J1, J3, J5, J6, J7. All other settings as in the main runs ($\\lambda_1 = 0.01$, 30 iterations). "
                  "Paired = episodes only $\\lambda_2{=}0.2$ solved : only the variant solved. $^\\dagger$Incomplete.",
                  "tab:prio_lam2")
    rep.append("tab_prio_lam2")


MID = [("midg_b1", "B1"), ("midg_pos", "W-IK pos"), ("midg_pose", "W-IK pose"), ("midg_prio", "Priority IK")]


def mid_onset(R, OUT, rep):
    """Joint locks 10 control steps after the first gripper-close command (mid-episode)."""
    recs = {m: {(t, j): M.load(os.path.join(R, SWEEP_DIR.format(m), M.fname("locked", t, j)))
                for t in M.TASKS for j in JR} for m, _ in MID}
    recs = {m: {k: v for k, v in r.items() if score(v) is not None} for m, r in recs.items()}
    if not recs["midg_prio"]:
        return
    P = recs["midg_prio"]
    rows, steps, n_ep, n_on = [], [], 0, 0
    for m, lab in MID:
        r = recs[m]
        if not r:
            continue
        common = {k: v for k, v in r.items() if k in P}
        perj = [M.r2(np.mean(v)) if (v := [r[k]["score"] for k in r if k[1] == j]) else "--" for j in JR]
        row = [lab + ("" if len(r) == 20 else f"$^\\dagger$ ({len(r)}/20)")] + perj + \
              [M.r2(np.mean([v["score"] for v in r.values()]))]
        if m == "midg_prio":
            row.append("--")
        else:
            b, c = pooled({k: P[k] for k in common}, common)
            row.append(f"{b}:{c} ({M.fmt_p(M.mcnemar(b, c))})")
        rows.append(row)
        if m == "midg_b1":                     # onset is identical across methods (same samples before it)
            for v in r.values():
                for w in v["raw"].get("onset_steps_per_worker") or []:
                    if isinstance(w, list):
                        n_ep += len(w); on = [x for x in w if x is not None]; n_on += len(on); steps += on
    cap = ("Mid-episode onset: the joint runs healthy and locks 10 control steps (0.5~s) after the policy's first "
           "gripper-close command, at the angle it has at that moment; the correcting methods re-read the fault "
           "before every policy call, so the first chunk after the onset can be executed uncorrected. Locked "
           "J1, J3, J5, J6, J7, 4 tasks, same seeds. Paired = episodes only Priority IK solved : only the method solved.")
    if n_ep:
        cap += (f" The onset occurred in {n_on} of {n_ep} episodes (median step {int(np.median(steps))}"
                f", range {min(steps)}--{max(steps)}); in the others the gripper never closed and no fault occurred.")
    M.write_table(OUT, "tab_mid_onset", ["Method", "J1", "J3", "J5", "J6", "J7", "Mean", "Prio:method"], rows,
                  cap, "tab:mid_onset")
    rep.append("tab_mid_onset")


def _resid(rec):
    """(pos mm, rot deg, max |dq| rad) of the executed corrections, from the run's own log."""
    raw = (rec or {}).get("raw") or {}
    st = raw.get("ik_stats") or raw.get("prio_stats") or raw.get("stats") or {}
    if "pos_err_after_mean" in st:
        return 1000 * st["pos_err_after_mean"], np.degrees(st["rot_err_after_mean"]), st.get("dq_free_max_mean")
    if "pos_after" in st:
        return 1000 * st["pos_after"], np.degrees(st["rot_after"]), st.get("dq")
    return None


def residuals(R, data, OUT, rep):
    """Final end-effector residual of the executed correction (logged during the rollouts)."""
    groups = {"prox.": (1, 3), "dist.": (6, 7)}
    cfgs = [("W-IK", (w, rho, dm, it), d) for (w, rho, dm, it), d in _wik_dirs(R).items()]
    cfgs.sort(key=lambda c: (c[1][2] is not None, c[1][3] or 30, -c[1][1], c[1][0]))
    rows = []
    for kind, key, d in cfgs + [("Prio", None, None)]:
        if kind == "Prio":
            recs = {(t, j): data.get(("locked", t, j, "prio")) for t in M.TASKS for j in JR}
            lab = "Priority IK ($\\lambda_2$ 0.2)"
        else:
            w, rho, dm, it = key
            recs = {(t, j): M.load(os.path.join(R, d, M.fname("locked", t, j))) for t in M.TASKS for j in JR}
            lab = f"W-IK $w_r$ {w:g}, $\\rho$ {rho:g}" + (f", $10^{{-{dm}}}$" if dm else "") + (f", {it} it." if it else "")
        row = [lab]
        ok = False
        for g, js in groups.items():
            v = [_resid(recs[k]) for k in recs if k[1] in js and recs[k]]
            v = [x for x in v if x is not None]
            sc = [recs[k]["score"] for k in recs if k[1] in js and recs[k]]
            if v:
                ok = True
                row += [f"{np.mean([x[0] for x in v]):.1f}", f"{np.mean([x[1] for x in v]):.1f}",
                        f"{np.mean([x[2] for x in v if x[2] is not None]):.2f}" if any(x[2] is not None for x in v) else "--",
                        M.r2(np.mean(sc)) if sc else "--"]
            else:
                row += ["--"] * 4
        if ok:
            rows.append(row)
    if len(rows) < 2:
        return
    M.write_table(OUT, "tab_residuals", ["Method", "prox. pos [mm]", "rot [deg]", "max$|\\Delta q|$ [rad]", "success",
                                         "dist. pos [mm]", "rot [deg]", "max$|\\Delta q|$ [rad]", "success"], rows,
                  "End-effector residual left by the executed correction, logged during the locked rollouts (mean over "
                  "corrected waypoints, then over conditions): proximal = J1, J3; distal = J6, J7; 4 tasks. "
                  "max$|\\Delta q|$ = largest change of a healthy joint.", "tab:residuals")
    rep.append("tab_residuals")


def gee_check(data, OUT, rep):
    """Frequentist cross-check of the mixed model: logistic GEE, episodes clustered by condition
    (exchangeable working correlation, robust sandwich SE)."""
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        print(f"  GEE skipped ({e})"); return
    rows = []
    for level in ["all"] + LEVELS:
        lv = LEVELS if level == "all" else [level]
        for m, lab in [("b1", "B1"), ("eci", "E-C-I"), ("rg", "RG-DDPM"), ("pos", "W-IK pos"), ("pose", "W-IK pose")]:
            recs = []
            for L in lv:
                for t in M.TASKS:
                    for j in M.JOINTS:
                        P, Q = data.get((L, t, j, "prio")), data.get((L, t, j, m))
                        if not (P and Q and P["pe"] and Q["pe"]):
                            continue
                        c = f"{L}{t}{j}"
                        recs += [dict(y=int(v > .5), prio=1, cond=c) for v in P["pe"].values()]
                        recs += [dict(y=int(v > .5), prio=0, cond=c) for v in Q["pe"].values()]
            if len(recs) < 200:
                continue
            df = pd.DataFrame(recs)
            try:
                fit = smf.gee("y ~ prio", "cond", df, family=sm.families.Binomial(),
                              cov_struct=sm.cov_struct.Exchangeable()).fit()
            except Exception as e:
                print(f"  GEE {level} {m}: {e}"); continue
            b, se, p = fit.params["prio"], fit.bse["prio"], fit.pvalues["prio"]
            rows.append([level, lab, f"{b:+.2f}", f"{se:.2f}", f"[{b - 1.96 * se:+.2f}, {b + 1.96 * se:+.2f}]",
                         f"{np.exp(b):.1f}", M.fmt_p(max(p, 1e-300))])
    if rows:
        M.write_table(OUT, "tab_gee", ["Level", "vs", "log-odds (Prio)", "robust SE", "95\\% CI", "odds ratio", "$p$"],
                      rows, "Logistic GEE (success $\\sim$ method), episodes clustered by condition, exchangeable "
                      "working correlation, robust (sandwich) standard errors: a frequentist cross-check of the "
                      "variational mixed model.", "tab:gee")
        rep.append("tab_gee")


def _auc(score_pos, score_neg):
    """P(score of a positive > score of a negative), ties count half."""
    if not score_pos or not score_neg:
        return float("nan")
    a = np.array(score_pos)[:, None]; b = np.array(score_neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def diagnosis(R, data, layer1, OUT, rep, tip_m=0.10, thr_mm=10.0):
    """Policy-free diagnosis as a classifier: does the kinematic error separate conditions with
    success >= 0.5 from the rest? Position only, and a fingertip-point error that adds the lever
    arm of the orientation error (pos + tip_m * rot), which also covers J7."""
    if not os.path.exists(layer1):
        return
    import csv
    rows_in = list(csv.DictReader(open(layer1)))
    out = []
    for target, mkey, col_p, col_r in [("B1", "b1", "override_pos_mm", "override_rot_deg"),
                                        ("Priority IK", "prio", "ik_pos_mm", "ik_rot_deg")]:
        xs, ys = {"pos": [], "tip": []}, []
        for r in rows_in:
            rec = data.get((r["level"], r["task"], int(r["joint"][-1]), mkey))
            if not rec:
                continue
            pmm, rdeg = float(r[col_p]), float(r[col_r])
            xs["pos"].append(pmm); xs["tip"].append(pmm + 1000 * tip_m * np.radians(rdeg)); ys.append(rec["score"])
        ys = np.array(ys)
        for k, lab in [("pos", "position error"), ("tip", f"fingertip error ({int(tip_m * 100)} cm)")]:
            x = np.array(xs[k]); good = ys >= 0.5
            auc = _auc(list(-x[good]), list(-x[~good]))
            acc = float(np.mean((x <= thr_mm) == good))
            out.append([target, lab, len(ys), f"{M.spearman(x, ys):+.2f}", f"{auc:.2f}", f"{acc:.2f}",
                        f"{int(good.sum())}/{len(ys)}"])
    if not out:
        return
    M.write_table(OUT, "tab_diagnosis", ["Outcome", "predictor", "conds", "Spearman", "AUC", f"acc. at {thr_mm:g} mm",
                                         "success$\\geq$0.5"], out,
                  "Policy-free kinematic diagnosis as a pre-deployment classifier over the 112 conditions. B1 is "
                  "predicted from the error the fault causes (override); Priority IK from the error left after "
                  "retargeting (residual). AUC: probability that a condition with success $\\geq 0.5$ has the smaller "
                  "error. Fingertip error = position error + lever arm $\\times$ orientation error (in radians), "
                  "which makes the J7 (pure yaw) faults visible.", "tab:diagnosis")
    rep.append("tab_diagnosis")


def mild_report(R, data, layer1, OUT, rep):
    L1 = {}
    if os.path.exists(layer1):
        import csv
        for r in csv.DictReader(open(layer1)):
            L1[(r["level"], r["task"], int(r["joint"][-1]))] = r
    lines = ["# Mild range faults: where Priority IK loses to W-IK pos", "",
             "Per joint, mean over the 4 tasks. dq = mean largest change of a healthy joint in the executed "
             "corrections (from the rollout logs); residual = policy-free W-IK residual (Layer1); paired = episodes "
             "only Priority IK solved : only W-IK pos solved.", "",
             "| joint | B1 | W-IK pos | W-IK pose | Priority IK | paired Prio:pos | Prio dq | W-IK pos dq | residual mm |",
             "|---|---|---|---|---|---|---|---|---|"]
    for j in M.JOINTS:
        g = {m: [data.get(("mild", t, j, m)) for t in M.TASKS] for m in ("b1", "pos", "pose", "prio")}
        mean = lambda m: "–" if not all(g[m]) else f"{np.mean([x['score'] for x in g[m]]):.2f}"
        B = C = 0
        for P, Q in zip(g["prio"], g["pos"]):
            if P and Q and P["pe"] and Q["pe"]:
                b, c = M.paired(P["pe"], Q["pe"]); B += b; C += c
        dq = lambda m: [r[2] for r in (_resid(x) for x in g[m]) if r and r[2] is not None]
        res = [float(L1[("mild", t, j)]["ik_pos_mm"]) for t in M.TASKS if ("mild", t, j) in L1]
        f = lambda v: "–" if not v else f"{np.mean(v):.3f}"
        lines.append(f"| J{j} | {mean('b1')} | {mean('pos')} | {mean('pose')} | {mean('prio')} | {B}:{C} | "
                     f"{f(dq('prio'))} | {f(dq('pos'))} | {'–' if not res else f'{np.mean(res):.1f}'} |")
    lines += ["", "Reading guide: at the mild level the fault often binds only briefly and the healthy trajectory is "
              "nearly reachable. If Priority IK's dq is much larger than W-IK pos's on the joints where it loses, the "
              "loss comes from moving healthy joints far from the policy's posture to remove a small error (W-IK pos "
              "keeps the posture via rho); the posture term that hurts on locked distal faults helps here.", ""]
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    open(os.path.join(OUT, "analysis", "mild_level.md"), "w").write("\n".join(lines))
    rep.append("analysis/mild_level.md")


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
    prio_lam2(a.results, data, a.out, rep)
    mid_onset(a.results, a.out, rep)
    residuals(a.results, data, a.out, rep)
    if a.glmm:
        gee_check(data, a.out, rep)
    diagnosis(a.results, data, a.layer1, a.out, rep)
    mild_report(a.results, data, a.layer1, a.out, rep)
    print("wrote:", ", ".join(rep))


if __name__ == "__main__":
    main()
