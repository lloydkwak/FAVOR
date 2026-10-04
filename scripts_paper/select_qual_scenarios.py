"""
Pick episodes for the qualitative figures and the supplementary video from existing
locked-sweep results (no new experiments: the chosen seeds are re-run with rendering on,
and render_qualitative.py checks the outcome against these JSONs).

A seed qualifies for a scenario when Priority IK succeeds and B1 and the contrast
baseline(s) fail on that exact seed. Among those, representative seeds are preferred:
W-IK pos and W-IK pose succeed or fail there as they do on most seeds of the condition
(so a baseline that usually works is not shown failing).
  A  distal   (J6, J7)  contrast: W-IK pose, preferred Bowl-Stove J7
  B  proximal (J1, J3)  contrast: W-IK pos
  C  video only: another proximal fault where both W-IK settings usually fail, preferred
     Bowl-Stove J1 (B1 0.00, W-IK pos 0.00, W-IK pose 0.15, Priority IK 0.75)
  D  video only: an unrecoverable fault (J2 / J4 locked) on which every method fails,
     so the video also shows the limits of all inference-time corrections
Seeds from the earliest chunk are preferred: the re-run only needs chunks up to the seed's
(seeds 10000+5k .. 10000+5k+4 form chunk k).

Usage (host, repo root): python scripts_paper/select_qual_scenarios.py [--results results]
                          [--out results/qual/scenarios.json]
"""
import argparse, json, os

TASKS = ["alphabet_soup", "milk", "bowl_ramekin", "bowl_stove"]
DIRS = {"b1": "libero_fault_sweep_locked_b1", "pos": "libero_fault_sweep_locked_ik",
        "pose": "libero_fault_sweep_locked_ik_pose", "prio": "libero_fault_sweep_locked_prio",
        "rg": "libero_fault_sweep_locked_rg"}
SPECS = [
    dict(id="A_distal", joints=[7, 6], contrast=["pose"], prefer=("bowl_stove", 7), paper=True),
    dict(id="B_proximal", joints=[1, 3], contrast=["pos"], prefer=("alphabet_soup", 3), paper=True),
    dict(id="C_proximal", joints=[1, 3], contrast=["pos", "pose"], prefer=("bowl_stove", 1), paper=False),
    dict(id="D_unrecoverable", joints=[2, 4], contrast=None, prefer=None, paper=False,
         note="kinematically unrecoverable: no method succeeds"),
]
START, NENV = 10000, 5


def load(R, m, t, j):
    p = os.path.join(R, DIRS[m], f"{t}_robot0_joint{j}_locked_na.json")
    if not os.path.exists(p):
        return None
    pe = json.load(open(p))["per_episode"]
    return {int(k): float(v) for k, v in pe.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/qual/scenarios.json")
    a = ap.parse_args()
    R = a.results

    rows = []
    for t in TASKS:
        for j in range(1, 8):
            d = {m: load(R, m, t, j) for m in DIRS}
            if any(d[m] is None for m in ("b1", "pos", "pose", "prio")):
                continue
            seeds = sorted(d["prio"])
            mean = {m: (sum(v.values()) / len(v) if v else None) for m, v in d.items()}
            rows.append(dict(task=t, joint=j, seeds=seeds, ep=d, mean=mean))

    out, used = [], set()
    for sp in SPECS:
        cands = []
        for r in rows:
            if r["joint"] not in sp["joints"] or (r["task"], r["joint"]) in used:
                continue
            ep = r["ep"]
            if sp["contrast"] is None:                       # failure case: every method fails
                good = [s for s in r["seeds"] if all(ep[m][s] < 1 for m in ("b1", "pos", "pose", "prio"))]
                if good:
                    worst = max(r["mean"][m] for m in ("b1", "pos", "pose", "prio"))
                    cands.append((sp["prefer"] == (r["task"], r["joint"]), len(good), -worst, r, good, good))
                continue
            good = [s for s in r["seeds"] if ep["prio"][s] >= 1 and ep["b1"][s] < 1
                    and all(ep[c][s] < 1 for c in sp["contrast"])]
            # representative: W-IK pos / pose behave on this seed as they do on most seeds of the condition
            strict = [s for s in good if all((ep[m][s] >= 1) == (r["mean"][m] >= 0.5) for m in ("pos", "pose"))]
            if not good:
                continue
            gap = r["mean"]["prio"] - max(r["mean"][c] for c in sp["contrast"])
            pref = sp["prefer"] == (r["task"], r["joint"])
            cands.append((pref, len(strict), gap, r, good, strict))
        if not cands:
            print(f"[{sp['id']}] no qualifying condition"); continue
        cands.sort(key=lambda c: (c[0], round(c[2], 2), c[1]), reverse=True)
        print(f"\n[{sp['id']}] candidates (task, joint, n_rep, n_good, gap, means b1/pos/pose/prio):")
        for pref, ns, gap, r, good, strict in cands[:6]:
            m = r["mean"]
            print(f"  {r['task']:<14} J{r['joint']}  rep={ns:2d} good={len(good):2d} gap={gap:+.2f}  "
                  f"{m['b1']:.2f}/{m['pos']:.2f}/{m['pose']:.2f}/{m['prio']:.2f}{'  (preferred)' if pref else ''}")
        _, _, gap, r, good, strict = cands[0]
        pool = strict or good
        chunk = lambda s: (s - START) // NENV
        k = min(chunk(s) for s in pool)
        picks = [s for s in pool if chunk(s) == k][:2]
        used.add((r["task"], r["joint"]))
        ep = r["ep"]
        out.append(dict(id=sp["id"], task=r["task"], joint=r["joint"], seeds=picks,
                        n_test=NENV * (k + 1), contrast=sp["contrast"] or [],
                        paper=sp["paper"], note=sp.get("note", ""),
                        orig={s: {m: ep[m][s] for m in ep if ep[m] is not None and s in ep[m]} for s in picks},
                        cond_mean={m: v for m, v in r["mean"].items() if v is not None}))
        print(f"  -> {r['task']} J{r['joint']} seeds {picks} (re-run n_test={NENV * (k + 1)})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
