"""
Revision experiments (reviewer requests). Same seeds (10000-10019), runner config and
policy construction as the main locked sweep, so every result pairs episode by episode
with the existing B1 / W-IK / Priority IK files.

Usage: python run_libero_revision.py <method> <task>

method
  nofault            B1 without any fault (healthy reference), all tasks
  wik_wXXX_rYYY      W-IK with rot_weight = XXX/100 and posture weight rho = YYY/100,
                     e.g. wik_w030_r010 (w_r 0.30, rho 0.10), wik_w100_r000 (pose, rho 0)
  prio_rev_l001      orientation-first priority with the 2nd task (position) damped by 0.01
                     instead of 0.2, separating the order effect from the damping effect
  rg_prioint         RG-DDPM whose internal correction is Priority IK (instead of weighted
                     pose IK, no motion limit on the correction), Priority IK at execution

Locked faults on the joints that are not near zero for every method: J1, J3, J5, J6, J7.
Output: results/libero_fault_sweep_locked_<method>/<task>_robot0_joint<j>_locked_na.json
        results/libero_nofault/<task>.json
"""
import json, os, re, sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
import favor_fault_runner  # noqa: F401
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid_libero import N_TEST, TEST_START_SEED, TASKS

JOINTS = [1, 3, 5, 6, 7]
RES = "/workspace/results"


def make_policy(method, base, runner, joint_name):
    kw = dict(base_seed=42, env_ref=runner.env, fault_joint_name=joint_name,
              fault_type="locked", fault_severity=None)
    m = re.fullmatch(r"wik_w(\d{3})_r(\d{3})", method)
    if m:
        from native_joint_policy import NativeJointPolicy
        ov = {"rot_weight": int(m.group(1)) / 100.0, "reg": int(m.group(2)) / 100.0}
        from ik_redistribution import summarize_ik_log
        return NativeJointPolicy(base, mode="ik", ik_overrides=ov, **kw), summarize_ik_log
    if method == "prio_rev_l001":
        import native_joint_policy_prio as npp
        from ik_priority_rev import ik_priority_rev
        npp.ik_priority = ik_priority_rev
        return npp.PrioIKPolicy(base, lam2=0.01, **kw), npp.summarize_prio_log
    if method == "rg_prioint":
        import reach_guided as rgm
        from reach_guided_prio import reach_correct_prio
        rgm.reach_correct = reach_correct_prio
        import native_joint_policy_rg as npr
        from ik_priority import ik_priority

        def _exec_prio(kin, q_target, joint_idx, q_con, **_):
            return ik_priority(kin, q_target, joint_idx, q_con, lam2=0.2)
        npr.ik_redistribute = _exec_prio
        return npr.RGNativeJointPolicy(base, **kw), npr.summarize_rg_log
    raise ValueError(method)


def load_base(task):
    payload = torch.load(open(TASKS[task]["ckpt"], "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=f"{RES}/_rev_scratch_{task}")
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    p = ws.ema_model if cfg.training.use_ema else ws.model
    p.to(torch.device("cuda:0")); p.eval()
    return p, cfg


def runner_for(task, cfg, joint_name, fault_type, tag):
    return FaultRobomimicImageRunner(
        output_dir=f"{RES}/_rev_run_{tag}_{task}",
        dataset_path=TASKS[task]["dataset"], shape_meta=cfg.task.shape_meta,
        fault_joint_name=joint_name, fault_type=fault_type, fault_severity=None,
        n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
        max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        render_obs_key="agentview_image", abs_action=True,
        actuation_mode="joint", joint_kp=TASKS[task].get("joint_kp", 150))


def per_episode(log):
    return {k.replace("test/sim_max_reward_", ""): v for k, v in log.items()
            if k.startswith("test/sim_max_reward_")}


def main():
    method, task = sys.argv[1], sys.argv[2]
    assert task in TASKS, task
    base, cfg = load_base(task)
    if method == "nofault":
        out_dir = f"{RES}/libero_nofault"; os.makedirs(out_dir, exist_ok=True)
        out = f"{out_dir}/{task}.json"
        if os.path.exists(out):
            print(f"SKIP (exists): {out}", flush=True); return
        from native_joint_policy import NativeJointPolicy
        runner = runner_for(task, cfg, "robot0_joint1", None, "nofault")
        log = runner.run(NativeJointPolicy(base, base_seed=42, fault_spec=None))
        json.dump({"b1": log.get("test/mean_score"), "per_episode": per_episode(log), "task": task,
                   "fault_type": None, "n_test": N_TEST}, open(out, "w"), indent=2)
        print(f"RESULT {task}/nofault/b1: {log.get('test/mean_score')}", flush=True)
        return
    out_dir = f"{RES}/libero_fault_sweep_locked_{method}"; os.makedirs(out_dir, exist_ok=True)
    for j in JOINTS:
        jn = f"robot0_joint{j}"
        out = f"{out_dir}/{task}_{jn}_locked_na.json"
        if os.path.exists(out):
            print(f"SKIP (exists): {out}", flush=True); continue
        runner = runner_for(task, cfg, jn, "locked", method)
        policy, summarize = make_policy(method, base, runner, jn)
        log = runner.run(policy)
        try:
            stats = summarize(policy) if summarize else None
        except Exception as e:
            stats = {"summary_error": str(e)}
        json.dump({method: log.get("test/mean_score"), "per_episode": per_episode(log), "task": task,
                   "joint": jn, "fault_type": "locked", "level": None, "n_test": N_TEST,
                   "stats": stats}, open(out, "w"), indent=2)
        print(f"RESULT {task}/{jn}/locked/{method}: {log.get('test/mean_score')}", flush=True)
        runner.env.close()
    print(f"DONE {task} {method}", flush=True)


if __name__ == "__main__":
    main()
