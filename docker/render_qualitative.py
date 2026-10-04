"""
Re-run selected sweep episodes with rendering on (no new experiments).

Each (scenario, method) re-runs the sweep runner with the exact sweep configuration
(same seeds, n_envs=5, base_seed=42, same policy construction), truncated to the
chunks that contain the selected seeds. Policy noise is seeded per (chunk, call), so
the truncated run reproduces those chunks exactly. RenderRecorder writes per-episode
traces, mp4s and lossless keyframes; the re-run outcome of every episode is compared
with the original sweep JSON and written to verify.json.

Also renders the setup snapshots (initial scene of every task, joint anchors).

Usage (container):
  python render_qualitative.py run       [--scenarios /workspace/results/qual/scenarios.json]
                                          [--methods b1,pos,pose,prio] [--only A_distal]
  python render_qualitative.py snapshots [--seed 10000]
Outputs under /workspace/results/qual/.
"""
import argparse, json, os, pickle, sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch, dill, hydra
import favor_fault_runner  # noqa: F401
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid_libero import TASKS

RES = "/workspace/results"
QUAL = os.path.join(RES, "qual")
ORIG_DIR = {"b1": "libero_fault_sweep_locked_b1", "pos": "libero_fault_sweep_locked_ik",
            "pose": "libero_fault_sweep_locked_ik_pose", "prio": "libero_fault_sweep_locked_prio",
            "rg": "libero_fault_sweep_locked_rg"}
REC = dict(camera="agentview", width=480, height=480, png_stride=5, fps=20, crf=14)

_BASE = {}


def base_policy(task):
    if task not in _BASE:
        payload = torch.load(open(TASKS[task]["ckpt"], "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=f"{RES}/_qual_scratch_{task}")
        ws.load_payload(payload, exclude_keys=None, include_keys=None)
        p = ws.ema_model if cfg.training.use_ema else ws.model
        p.to(torch.device("cuda:0")); p.eval()
        _BASE.clear(); _BASE[task] = (p, cfg)
    return _BASE[task]


def make_runner(task, joint_name, cfg, n_test, record_cfg, out_tag):
    return FaultRobomimicImageRunner(
        output_dir=f"{RES}/_qual_run_{out_tag}",
        dataset_path=TASKS[task]["dataset"], shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name, fault_type="locked", fault_severity=None,
        n_train=0, n_test=n_test, test_start_seed=10000, n_envs=5,
        max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        render_obs_key="agentview_image", abs_action=True,
        actuation_mode="joint", joint_kp=TASKS[task].get("joint_kp", 150),
        record_cfg=record_cfg)


def make_policy(method, base, runner, joint_name):
    from native_joint_policy import NativeJointPolicy
    kw = dict(base_seed=42, env_ref=runner.env, fault_joint_name=joint_name,
              fault_type="locked", fault_severity=None)
    if method == "b1":
        return NativeJointPolicy(base, base_seed=42, fault_spec=None)
    if method == "pos":
        return NativeJointPolicy(base, mode="ik", **kw)
    if method == "pose":
        return NativeJointPolicy(base, mode="ik", ik_overrides={"rot_weight": 1.0}, **kw)
    if method == "prio":
        from native_joint_policy_prio import PrioIKPolicy
        return PrioIKPolicy(base, lam2=0.2, **kw)
    if method == "rg":
        from native_joint_policy_rg import RGNativeJointPolicy
        return RGNativeJointPolicy(base, **kw)
    raise ValueError(method)


def orig_episodes(method, task, j):
    p = os.path.join(RES, ORIG_DIR[method], f"{task}_robot0_joint{j}_locked_na.json")
    if not os.path.exists(p):
        return None
    return {int(k): float(v) for k, v in json.load(open(p))["per_episode"].items()}


def cmd_run(a):
    scen = json.load(open(a.scenarios))
    methods = a.methods.split(",")
    vpath = os.path.join(QUAL, "verify.json")
    verify = json.load(open(vpath)) if os.path.exists(vpath) else {}
    for sc in scen:
        if a.only and sc["id"] not in a.only.split(","):
            continue
        task, j = sc["task"], sc["joint"]
        jn = f"robot0_joint{j}"
        for m in methods:
            tag = f"{sc['id']}/{m}"
            key = f"{sc['id']}:{m}"
            if key in verify and not a.force:
                print(f"SKIP (done) {tag}", flush=True); continue
            base, cfg = base_policy(task)
            rec = dict(REC, out_dir=QUAL, tag=tag, record_seeds=sc["seeds"], camera=sc.get("camera", REC["camera"]))
            runner = make_runner(task, jn, cfg, sc["n_test"], rec, f"{sc['id']}_{m}")
            policy = make_policy(m, base, runner, jn)
            log = runner.run(policy)
            try:
                runner.env.call("flush")
            except Exception as e:
                print(f"  flush rpc failed ({e}); relying on the final reset", flush=True)
            new = {int(k.replace("test/sim_max_reward_", "")): float(v)
                   for k, v in log.items() if k.startswith("test/sim_max_reward_")}
            old = orig_episodes(m, task, j) or {}
            same = {s: (s in old and old[s] == new[s]) for s in new}
            verify[key] = dict(task=task, joint=j, method=m, seeds=sc["seeds"], rerun=new,
                               orig={s: old.get(s) for s in new},
                               match_all=all(same.values()),
                               match_selected=all(same.get(s, False) for s in sc["seeds"]))
            json.dump(verify, open(vpath, "w"), indent=2)
            sel = {s: (old.get(s), new.get(s)) for s in sc["seeds"]}
            print(f"RERUN {tag}: match {sum(same.values())}/{len(same)}  selected (orig, rerun) {sel}", flush=True)
            runner.env.close()
            del runner, policy
    print("DONE run", flush=True)


def cmd_snapshots(a):
    out = {}
    for task in TASKS:
        _, cfg = base_policy(task)
        rec = dict(REC, out_dir=QUAL, tag=f"snap/{task}", record_seeds=[])
        runner = make_runner(task, "robot0_joint4", cfg, 5, rec, f"snap_{task}")
        env = runner.env
        env.call_each("run_dill_function", args_list=[(x,) for x in runner.env_init_fn_dills[:5]])
        env.reset()
        i = a.seed - 10000
        snaps = env.call("snapshot", highlight_joint=None)
        hl = {j: env.call("snapshot", highlight_joint=f"robot0_joint{j}")[i]["img"] for j in range(1, 8)}
        out[task] = dict(plain=snaps[i], highlight=hl, seed=a.seed)
        env.close()
        print(f"SNAP {task}: cameras {snaps[i]['cameras']}", flush=True)
    os.makedirs(QUAL, exist_ok=True)
    with open(os.path.join(QUAL, "snapshots.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("DONE snapshots", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "snapshots"])
    ap.add_argument("--scenarios", default=os.path.join(QUAL, "scenarios.json"))
    ap.add_argument("--methods", default="b1,pos,pose,prio")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=10000)
    a = ap.parse_args()
    {"run": cmd_run, "snapshots": cmd_snapshots}[a.cmd](a)
