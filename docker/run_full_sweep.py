"""
Full statistical sweep: B1 (unconstrained terminal IK) vs FAVOR (fault-aware
terminal IK, post-hoc single-shot projection -- confirmed this session to
match full mid-denoising ECI exactly, 0.85=0.85), actuation_mode='joint'.

Covers: nominal (no fault, 1 unit per task) + full fault grid (7 joints x 5
fault conditions x 3 tasks = 105 units) = 108 total units.

In the nominal unit, B1 and FAVOR are the IDENTICAL code path (fault_spec=None
for both) -- run anyway for pipeline uniformity (same per-seed JSON structure
for every condition, simpler downstream aggregation) and as a live consistency
check (should always match).

CRITICAL (fixed this session after catching a real bug): the environment's
physical fault_type must be the SAME real fault for both b1 and favor in
fault units -- only fault_spec (whether the policy/IK is TOLD about it)
differs. Passing fault_type=None for b1 would silently remove the fault from
b1's environment too, invalidating the whole comparison.

Each of the 108 units is an independent, resumable file:
results/sweep_full/{task}_{joint}_{fault}_{severity}.json
Already-completed units are skipped on restart -- safe to run unattended
across GPU crashes / restarts (this session hit a hardware Xid 79 crash once
already).
"""
import sys, os, json, time
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid import JOINTS, FAULT_CONDITIONS, TASKS, N_TEST, TEST_START_SEED

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
N_ENVS = 28
OUT_DIR = "/workspace/results/sweep_full"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_STEPS = 400
N_OBS_STEPS = 2
N_ACTION_STEPS = 8
RENDER_KEY = 'agentview_image'

# CRITICAL FIX (found after Can/Square scored 0.0 across the board): each
# task's shape_meta must come from ITS OWN checkpoint's cfg.task.shape_meta,
# NOT a single hardcoded spec copied from Lift. Can/Square's demonstrations
# include an 'object' observation key that Lift's policy was trained
# without -- omitting it silently blinds the Can/Square policy to the
# object's position entirely, causing complete task failure regardless of
# any fault/correction logic (confirmed: both b1 AND favor scored exactly
# 0.000 identically, which is the signature of "policy never sees the
# object" rather than a fault-related effect).
_ckpt_cache = {}
_shape_meta_cache = {}
def get_policy(task_name):
    if task_name not in _ckpt_cache:
        ckpt = TASKS[task_name]["ckpt"]
        payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_scratch_sweep_{task_name}")
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        base_policy.to(torch.device("cuda:0"))
        base_policy.eval()
        _ckpt_cache[task_name] = base_policy
        _shape_meta_cache[task_name] = cfg.task.shape_meta
    return _ckpt_cache[task_name]

def get_shape_meta(task_name):
    if task_name not in _shape_meta_cache:
        get_policy(task_name)  # populates the cache as a side effect
    return _shape_meta_cache[task_name]

def run_batch(task_name, joint_name, real_fault_type, severity, fault_spec_or_none, batch_start_seed, batch_n):
    base_policy = get_policy(task_name)
    shape_meta = get_shape_meta(task_name)
    runner = FaultRobomimicImageRunner(
        output_dir="/workspace/results/_sweep_scratch_out",
        dataset_path=TASKS[task_name]["dataset"], shape_meta=shape_meta,
        fault_joint_name=joint_name if joint_name != "none" else "robot0_joint1",
        fault_type=real_fault_type, fault_severity=severity,
        n_train=0, n_test=batch_n, test_start_seed=batch_start_seed, n_envs=batch_n,
        max_steps=MAX_STEPS, n_obs_steps=N_OBS_STEPS, n_action_steps=N_ACTION_STEPS,
        render_obs_key=RENDER_KEY, abs_action=True,
        actuation_mode='joint',
    )
    favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec_or_none, env_ref=runner.env,
                                    actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)
    log = runner.run(favor)
    per_seed = {}
    for seed_offset in range(batch_n):
        seed = batch_start_seed + seed_offset
        key = f"test/sim_max_reward_{seed}"
        if key in log:
            per_seed[seed] = log[key]
    return per_seed

def run_condition(task_name, joint_name, real_fault_type, severity, joint_idx):
    fault_spec = None
    if real_fault_type is not None:
        fault_spec = {'joint_idx': joint_idx, 'q_lock': 0.0, 'fault_type': real_fault_type,
                      'q_lo': q_lo, 'q_hi': q_hi}

    results = {}
    for label, fs in [("b1", None), ("favor", fault_spec)]:
        per_seed_all = {}
        remaining, start = N_TEST, TEST_START_SEED
        while remaining > 0:
            batch_n = min(N_ENVS, remaining)
            per_seed = run_batch(task_name, joint_name, real_fault_type, severity, fs, start, batch_n)
            per_seed_all.update(per_seed)
            start += batch_n
            remaining -= batch_n
        results[label] = per_seed_all
    return results

def condition_filename(task_name, joint_name, fault_type, severity):
    ft_str = "nominal" if fault_type is None else fault_type
    sev_str = "na" if severity is None else str(severity).replace(".", "p")
    return f"{OUT_DIR}/{task_name}_{joint_name}_{ft_str}_{sev_str}.json"

def build_units():
    units = []
    for task_name in TASKS:
        units.append((task_name, "none", None, None, 0))
    for task_name in TASKS:
        for joint_idx, joint_name in enumerate(JOINTS):
            for fault_type, severity in FAULT_CONDITIONS:
                units.append((task_name, joint_name, fault_type, severity, joint_idx))
    return units

def main():
    units = build_units()
    total = len(units)
    print(f"Total units: {total} (3 nominal + {total-3} fault conditions)")
    done = 0
    for task_name, joint_name, fault_type, severity, joint_idx in units:
        fname = condition_filename(task_name, joint_name, fault_type, severity)
        done += 1
        if os.path.exists(fname):
            print(f"[{done}/{total}] SKIP (already done): {fname}")
            continue
        print(f"[{done}/{total}] RUNNING: task={task_name} joint={joint_name} fault={fault_type} severity={severity}", flush=True)
        t0 = time.time()
        try:
            results = run_condition(task_name, joint_name, fault_type, severity, joint_idx)
            results["_meta"] = {
                "task": task_name, "joint": joint_name, "fault_type": fault_type,
                "severity": severity, "n_test": N_TEST, "elapsed_sec": time.time() - t0
            }
            with open(fname, "w") as f:
                json.dump(results, f, indent=2)
            b1_mean = sum(results["b1"].values()) / len(results["b1"])
            favor_mean = sum(results["favor"].values()) / len(results["favor"])
            print(f"  -> b1_mean={b1_mean:.3f} favor_mean={favor_mean:.3f} elapsed={time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  !! FAILED: {e}", flush=True)
            with open(fname + ".FAILED", "w") as f:
                f.write(str(e))
            raise
    print("ALL UNITS COMPLETE" if done == total else "STOPPED EARLY")

if __name__ == "__main__":
    main()
