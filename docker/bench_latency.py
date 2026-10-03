"""
Inference latency per policy call (one call = one 16-step chunk for 5 parallel envs, of
which 8 steps are executed), measured inside real rollouts.

For each method the sweep runner is run on one locked condition (default Bowl-Stove J7,
seeds 10000-10004) and every predict_action call is timed with CUDA synchronization.
The first call of each run (CUDA warm-up) is dropped.

Usage: python bench_latency.py [task] [joint]
Output: results/latency/<task>_J<joint>.json and a printed table
"""
import json, os, sys, time
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch
from render_qualitative import base_policy, make_runner, make_policy

METHODS = ["b1", "pos", "pose", "prio", "rg"]


def timed(policy, log):
    orig = policy.predict_action

    def wrapped(obs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig(obs)
        torch.cuda.synchronize(); log.append(time.perf_counter() - t0)
        return out
    policy.predict_action = wrapped
    return policy


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "bowl_stove"
    j = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    base, cfg = base_policy(task)
    res = {}
    for m in METHODS:
        runner = make_runner(task, f"robot0_joint{j}", cfg, 5, None, f"lat_{m}")
        log = []
        runner.run(timed(make_policy(m, base, runner, f"robot0_joint{j}"), log))
        runner.env.close()
        t = np.array(log[1:]) * 1000
        res[m] = dict(calls=len(t), mean_ms=float(t.mean()), std_ms=float(t.std()),
                      median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)))
        print(f"LAT {m:5s} calls={len(t):4d} mean={t.mean():7.1f} ms  median={np.median(t):7.1f}  p95={np.percentile(t, 95):7.1f}",
              flush=True)
    b = res["b1"]["mean_ms"]
    for m in METHODS:
        res[m]["overhead_vs_b1_ms"] = res[m]["mean_ms"] - b
    gpu = torch.cuda.get_device_name(0)
    os.makedirs("/workspace/results/latency", exist_ok=True)
    json.dump({"task": task, "joint": j, "gpu": gpu, "batch_envs": 5, "methods": res},
              open(f"/workspace/results/latency/{task}_J{j}.json", "w"), indent=2)
    print(f"DONE latency ({gpu})", flush=True)


if __name__ == "__main__":
    main()
