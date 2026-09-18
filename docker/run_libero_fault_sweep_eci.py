"""
E-C-I sweep on LIBERO (Franka Panda, 7-DOF): SAME 21 conditions, SAME
seeds (TEST_START_SEED=10000, n_test=20) as B1/select/random_n
(sweep_grid_libero.py), so this is a genuine same-platform, same-seed
comparison -- unlike the project's earlier E-C-I evidence, which came
from two DIFFERENT platforms tested at different times:

  - Lift/Can/Square (robosuite, Franka Panda, 7-DOF): E-C-I showed real
    gains (Lift joint1 +0.60, joint3 +0.18, joint5 +0.12, joint6 +0.14;
    Can joint6 +0.42) but also one regression (Can joint7 -0.04) and was
    NEVER validated with a paired significance test (McNemar) at the
    time -- this was flagged explicitly in that session's own analysis.
  - RoboTwin (aloha, 6-DOF): E-C-I showed 0/8 improvement, consistent
    with the self-motion manifold argument (dim M = n-6 = 0 for 6-DOF,
    so no joint has room to redistribute into regardless of method).

Neither of those is a controlled head-to-head against Select, which was
only ever run on LIBERO. This script closes that gap: same policy
checkpoints, same fault conditions, same seeds, same per-episode logging
(for McNemar) as the existing B1/select/random_n LIBERO sweeps, so a
direct 4-way comparison (B1 / E-C-I / Select / Random-N) becomes possible
on one platform.

mode='eci' in NativeJointPolicy already implements this via
joint_eci_projector.eci_conditional_sample (PPR: predict-project-renoise
at every denoising step, projecting ALL 7 joints -- identity for healthy
joints, narrowed range for the faulted one) -- no new mechanism, just a
new grid to run it against.

Usage: python run_libero_fault_sweep_eci.py <task_name>
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid_libero import JOINTS, FAULT_CONDITIONS, N_TEST, TEST_START_SEED, TASKS

OUT_DIR = "/workspace/results/libero_fault_sweep_eci"
os.makedirs(OUT_DIR, exist_ok=True)

task_name = sys.argv[1]
assert task_name in TASKS, f"unknown task {task_name!r}, expected one of {list(TASKS)}"

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_sweep_scratch_eci_{task_name}")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

for joint_name in JOINTS:
    for fault_type, severity in FAULT_CONDITIONS:
        fname = f"{task_name}_{joint_name}_{fault_type}_{severity}.json".replace("None", "na")
        out_path = os.path.join(OUT_DIR, fname)
        if os.path.exists(out_path):
            print(f"SKIP (exists): {fname}")
            continue

        runner = FaultRobomimicImageRunner(
            output_dir=f"/workspace/results/_sweep_run_eci_{task_name}",
            dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
            n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key="agentview_image", abs_action=True,
            actuation_mode="joint", joint_kp=150,
        )
        policy = NativeJointPolicy(
            base_policy, base_seed=42, mode='eci', env_ref=runner.env,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
        )
        log = runner.run(policy)
        score = log.get("test/mean_score")
        per_episode = {k.replace("test/sim_max_reward_", ""): v
                       for k, v in log.items() if k.startswith("test/sim_max_reward_")}
        print(f"  {task_name}/{joint_name}/{fault_type}/{severity}/eci: {score}")

        with open(out_path, "w") as f:
            json.dump({"eci": score, "per_episode": per_episode}, f, indent=2)
        print(f"SAVED: {fname}")

print(f"TASK {task_name} ECI SWEEP COMPLETE")
