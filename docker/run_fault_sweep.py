"""
Phase 3 full sweep: for one task, run baseline + all (joint, fault_type,
severity) conditions from sweep_grid.py at n_test=50, same test_start_seed
throughout (paired seeds -> valid for McNemar later).

Usage inside container:
    python docker/run_fault_sweep.py --task lift
"""
import sys, os, json, argparse, time
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid import JOINTS, FAULT_CONDITIONS, TASKS, N_TEST, TEST_START_SEED

def load_policy(ckpt_path):
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device("cuda:0"))
    policy.eval()
    return policy, cfg

def run_one(policy, cfg, dataset_path, output_dir, joint_name, fault_type, severity):
    os.makedirs(output_dir, exist_ok=True)
    runner = FaultRobomimicImageRunner(
        output_dir=output_dir,
        dataset_path=dataset_path,
        shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name,
        fault_type=fault_type,
        fault_severity=severity,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED,
        n_envs=min(28, N_TEST),
        max_steps=cfg.task.env_runner.max_steps,
        n_obs_steps=cfg.task.env_runner.n_obs_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        render_obs_key=cfg.task.env_runner.render_obs_key,
        abs_action=cfg.task.env_runner.abs_action,
        n_test_vis=0,
    )
    log = runner.run(policy)
    with open(os.path.join(output_dir, "eval_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str, sort_keys=True)
    return log.get("test/mean_score")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    args = ap.parse_args()

    task = args.task
    ckpt = TASKS[task]["ckpt"]
    dataset = TASKS[task]["dataset"]
    policy, cfg = load_policy(ckpt)

    summary = {}
    base_dir = f"/workspace/results/phase3/{task}"
    os.makedirs(base_dir, exist_ok=True)

    # baseline (no fault) — joint_name is irrelevant when fault_type=None
    t0 = time.time()
    score = run_one(policy, cfg, dataset, f"{base_dir}/baseline", "robot0_joint1", None, None)
    print(f"[{task}] baseline -> {score}  ({time.time()-t0:.0f}s)")
    summary["baseline"] = score

    for joint in JOINTS:
        for fault_type, severity in FAULT_CONDITIONS:
            key = f"{joint}__{fault_type}__{severity}"
            out_dir = f"{base_dir}/{key}"
            t0 = time.time()
            score = run_one(policy, cfg, dataset, out_dir, joint, fault_type, severity)
            dt = time.time() - t0
            print(f"[{task}] {key} -> {score}  ({dt:.0f}s)")
            summary[key] = score
            with open(f"{base_dir}/_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    print("ALL DONE:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
