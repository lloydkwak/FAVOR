"""
Export every per-episode outcome under results/ to one tracked CSV, so that the paired
statistics (McNemar counts, condition-level tests) can be re-derived without the raw
result files. Also copies the policy-free kinematic analysis (layer1_v2.csv).

Rows: sweep (result directory), task, joint, fault_type, level, seed, success
  level: mild / moderate / severe for range faults (from the file name), locked, or none
Usage (host, repo root): python scripts_paper/export_episodes.py [--results results] [--out paper/data]
"""
import argparse, csv, glob, json, os, re, shutil

LEVEL_RE = re.compile(r"_range_reduced_(mild|moderate|severe)\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/data")
    ap.add_argument("--layer1", default="analysis_out/layer1_v2.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, n_files = [], 0
    for d in sorted(glob.glob(os.path.join(a.results, "libero_*"))):
        if not os.path.isdir(d):
            continue
        sweep = os.path.basename(d)
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                js = json.load(open(p))
            except Exception:
                continue
            pe = js.get("per_episode")
            if not pe:
                continue
            n_files += 1
            name = os.path.basename(p)
            m = LEVEL_RE.search(name)
            ft = js.get("fault_type") or ("locked" if "_locked_" in name else ("range_reduced" if m else "none"))
            level = m.group(1) if m else (js.get("level") or ("locked" if ft == "locked" else "none"))
            joint = js.get("joint") or ""
            jm = re.search(r"joint(\d)", joint or name)
            task = js.get("task") or name.split("_robot0")[0].replace(".json", "")
            for seed, v in sorted(pe.items(), key=lambda kv: int(kv[0])):
                rows.append([sweep, task, int(jm.group(1)) if jm else "", ft, level, int(seed), int(float(v) > 0.5)])
    out = os.path.join(a.out, "episodes.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "task", "joint", "fault_type", "level", "seed", "success"])
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} episodes from {n_files} files")
    if os.path.exists(a.layer1):
        shutil.copy(a.layer1, os.path.join(a.out, "layer1_v2.csv"))
        print(f"copied {a.layer1} -> {a.out}/layer1_v2.csv")
    else:
        print(f"layer1 csv not found at {a.layer1} (skipped)")


if __name__ == "__main__":
    main()
