"""
LIBERO Phase 2 fault sweep: mode='select' only (Random-N and select_comp
deferred -- see project notes), same 21 conditions as Phase 1
(sweep_grid_libero.py: 7 joints x locked x 3 tasks), SAME seeds
(TEST_START_SEED=10000) for valid paired comparison against the
fault_sweep_libero_phase1_results.csv B1 numbers.

n_select=8, n_envs=2 (batch=16 for the diffusion call) -- chosen for GPU
memory: n_envs=5 with n_select=8 (batch=40) hit CUDA OOM in
test_select_mode_fault.py; n_envs=2 leaves headroom. n_test=20 is reached
over multiple reset rounds within the runner (FaultRobomimicImageRunner
handles this internally via n_envs < n_test).

Usage: python run_libero_fault_sweep_phase2_select.py <task_name>
"""
import sys, os, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from sweep_grid_libero import JOINTS, FAULT_CONDITIONS, N_TEST, TEST_START_SEED, TASKS

OUT_DIR = "/workspace/results/libero_fault_sweep_phase2_select"
os.makedirs(OUT_DIR, exist_ok=True)

task_name = sys.argv[1]
assert task_name in TASKS, f"unknown task {task_name!r}, expected one of {list(TASKS)}"

payload = torch.load(open(TASKS[task_name]["ckpt"], "rb"), pickle_module=dill)
cfg = payload["cfg"]
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_sweep_scratch_select_{task_name}")
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
            output_dir=f"/workspace/results/_sweep_run_select_{task_name}",
            dataset_path=TASKS[task_name]["dataset"], shape_meta=cfg.task.shape_meta,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
            n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=5,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key="agentview_image", abs_action=True,
            actuation_mode="joint", joint_kp=150,
        )
        # n_select=32, chunk_size=8: chunked sampling keeps GPU memory flat
        # regardless of n_envs (measured: n_envs=1/2/5 all peak ~1.22GB at
        # chunk_size=8), so n_envs=5 is used for sweep speed (4 reset
        # rounds instead of 10) while N=32 closes to within 1.1-1.2x of
        # N=64's achievable minimum epsilon (N=8's single-shot batch was
        # 1.7-2.2x worse -- see check_n_vs_min_epsilon.py).
        policy = NativeJointPolicy(
            base_policy, base_seed=42, mode='select', env_ref=runner.env,
            fault_joint_name=joint_name, fault_type=fault_type, fault_severity=severity,
            n_select=32, chunk_size=8,
        )
        log = runner.run(policy)
        score = log.get("test/mean_score")
        per_episode = {k.replace("test/sim_max_reward_", ""): v
                       for k, v in log.items() if k.startswith("test/sim_max_reward_")}
        print(f"  {task_name}/{joint_name}/{fault_type}/{severity}/select: {score}")

        with open(out_path, "w") as f:
            json.dump({"select": score, "per_episode": per_episode}, f, indent=2)
        print(f"SAVED: {fname}")

print(f"TASK {task_name} SELECT SWEEP COMPLETE")
