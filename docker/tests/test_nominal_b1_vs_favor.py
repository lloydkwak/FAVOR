"""
Nominal (no real physical constraint) evaluation across all 3 tasks:
  - B1: fault_spec=None, pure original predict_action
  - FAVOR-eci: fault_spec set to a VACUOUS constraint (range_reduced with
    the FULL physical joint range, i.e. no actual narrowing) -- this
    exercises eci_conditional_sample's actual project+renoise code path
    (project_fault becomes a no-op numerically, but the sampling formula
    itself, including the renoise step, is genuinely different from the
    original conditional_sample) without imposing any real task difficulty.
    Confirms the custom sampling code doesn't degrade performance when the
    constraint is non-binding -- the real regression check this session
    needed (the earlier "exact numerical match" test was theoretically
    flawed, per this session's analysis).
  - FAVOR-posthoc: same vacuous constraint, single terminal projection.

n_test=10 for a fast first pass (matches training's own rollout scale);
rerun with n_test=28 (project standard) once this confirms the pipeline
works end-to-end.
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner

TASKS = {
    "square": {
        "ckpt": "/workspace/data/outputs/2026.08.10/05.53.47_train_diffusion_unet_hybrid_square_image_joint/checkpoints/epoch=0345-test_mean_score=0.800.ckpt",
        "dataset": "/workspace/diffusion_policy/data/robomimic/datasets/square/ph/image_abs.hdf5",
    },
    "can": {
        "ckpt": "/workspace/data/outputs/joint_train_can_run/checkpoints/epoch=0165-test_mean_score=1.000.ckpt",
        "dataset": "/workspace/diffusion_policy/data/robomimic/datasets/can/ph/image_abs.hdf5",
    },
    "lift": {
        "ckpt": "/workspace/data/outputs/joint_train_lift_run/checkpoints/epoch=0055-test_mean_score=1.000.ckpt",
        "dataset": "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5",
    },
}

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
N_TEST = 50
N_ENVS = 25  # process in 2 chunks of 25 (FaultRobomimicImageRunner.run()
             # already chunks n_test/n_envs internally via math.ceil --
             # same pattern as the project's official sweep scripts), so
             # only 25 sim environments are ever resident in GPU/CPU memory
             # at once, while still evaluating a full n=50 for statistics.

results = {}
for task_name, info in TASKS.items():
    payload = torch.load(open(info["ckpt"], 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_scratch_nominal_{task_name}")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    base_policy.to(torch.device("cuda:0"))
    base_policy.eval()

    vacuous_fault = {'q_lo': q_lo, 'q_hi': q_hi}  # full physical range on all 7 joints -> identity

    task_results = {}
    # B1 and FAVOR-posthoc are SKIPPED this run: both are mathematically
    # unaffected by today's all-joints redesign (B1 never touches
    # project_fault at all; posthoc's vacuous-fault clip is a proven
    # identity operation under full physical range on every joint, matching
    # this session's earlier n=50 confirmation of exact B1==posthoc match).
    # Only favor_eci exercises materially different code (project_fault now
    # runs on all 7 joints per denoising step instead of 1), so only it is
    # re-verified here.
    for label, kwargs in [
        ("favor_eci", dict(fault_spec=vacuous_fault, mode='eci')),
    ]:
        kwargs = dict(kwargs, base_seed=42)
        runner = FaultRobomimicImageRunner(
            output_dir=f"/workspace/results/_nominal_{task_name}_{label}_out",
            dataset_path=info["dataset"], shape_meta=cfg.task.shape_meta,
            fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
            n_train=0, n_test=N_TEST, test_start_seed=10000, n_envs=N_ENVS,
            max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
            render_obs_key='agentview_image', abs_action=True,
            actuation_mode='joint',
        )
        policy = NativeJointPolicy(base_policy, **kwargs)
        log = runner.run(policy)
        score = log.get("test/mean_score")
        task_results[label] = score
        print(f"{task_name} / {label}: score = {score}")
    results[task_name] = task_results

print()
print("=== SUMMARY (n_test=%d) ===" % N_TEST)
for task_name, task_results in results.items():
    print(task_name, task_results)

import json
out_path = "/workspace/results/nominal_b1_vs_favor_n50.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved to {out_path}")
