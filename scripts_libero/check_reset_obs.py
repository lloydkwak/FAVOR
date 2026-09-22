"""
bowl_stove/drawer closed-loop failure diagnosis.

Known facts (from prior sessions): demo-action replay succeeds (predicate
fires, reward=1.0), and the trained policy's own open-loop predictions
(fed the demo's own recorded observations) match the demo's actions
closely. But closed-loop rollout (policy's own actions feed back into the
env to produce the next observation) yields reward=0 throughout, for both
tasks, despite the same pipeline that got alphabet_soup/milk/bowl_ramekin
working.

This script checks the two remaining candidate causes, which produce the
same symptom (works when checked individually, fails end-to-end):

  (A) env.reset() initial state mismatch: does live reset() put the robot/
      scene in a state that matches demo_0's t=0 state closely, or does it
      differ (e.g. object not settled, camera not warmed up, wrong initial
      joint config)? If policy has never seen this initial state, its
      first action(s) could already be bad, and with no correction
      mechanism the error compounds.

  (B) Closed-loop divergence point: run an actual policy rollout and
      compare the ACHIEVED joint trajectory against the demo trajectory,
      step by step. If divergence starts immediately (step 0-5), that
      points to (A). If it starts gradually partway through, that points
      to ordinary closed-loop compounding error (which would then need a
      different fix -- e.g. this task's geometry being less forgiving of
      small errors than pick-place bowl tasks).

Run for both bowl_stove and drawer.
"""
import sys, os
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import h5py
import numpy as np
import torch, dill, hydra

from favor_fault_runner import FaultRobomimicImageRunner

TASKS = {
    "bowl_stove": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_bowl_stove/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_bowl_stove/ph/image_abs.hdf5",
    },
    "drawer": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_drawer/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_drawer/ph/image_abs.hdf5",
    },
    "alphabet_soup": {
        "ckpt": "/workspace/data/outputs/joint_train_libero_alphabet_soup/checkpoints/latest.ckpt",
        "dataset": "/workspace/data/robomimic/datasets/libero_alphabet_soup/ph/image_abs.hdf5",
    },
}


def check_task(task_name, info):
    print(f"\n{'='*60}\n{task_name}\n{'='*60}")

    with h5py.File(info["dataset"], "r") as f:
        demo_q = f["data/demo_0/obs/robot0_joint_pos"][:]
        demo_eef_pos = f["data/demo_0/obs/robot0_eef_pos"][:]
        demo_gripper = f["data/demo_0/obs/robot0_gripper_qpos"][:]

    print(f"demo_0 t=0 joint_pos: {np.round(demo_q[0], 4)}")
    print(f"demo_0 t=0 eef_pos:   {np.round(demo_eef_pos[0], 4)}")
    print(f"demo_0 t=0 gripper:   {np.round(demo_gripper[0], 4)}")

    # --- (A) live reset() comparison ---
    payload = torch.load(open(info["ckpt"], "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_scratch_reset_check_{task_name}")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    base_policy.to(torch.device("cuda:0"))
    base_policy.eval()

    runner = FaultRobomimicImageRunner(
        output_dir=f"/workspace/results/_reset_check_out_{task_name}",
        dataset_path=info["dataset"], shape_meta=cfg.task.shape_meta,
        fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
        n_train=0, n_test=5, test_start_seed=10000, n_envs=5,
        max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        render_obs_key="agentview_image", abs_action=True,
        actuation_mode="joint", joint_kp=150,
    )

    env = runner.env.env_fns[0]()
    obs = env.reset()
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    addrs = [raw.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1, 8)]
    q_now = np.array([raw.sim.data.qpos[a] for a in addrs])
    print(f"\nlive reset() joint_pos: {np.round(q_now, 4)}")
    print(f"|reset - demo_t0| joint diff: {np.round(np.abs(q_now - demo_q[0]), 4)}")
    print(f"  L2 norm: {np.linalg.norm(q_now - demo_q[0]):.4f}  "
          f"(compare: mean per-step demo motion is typically ~0.01-0.04)")

    # --- (B) closed-loop rollout divergence point ---
    class NativeJointPolicyAdapter:
        def __init__(self, base_policy):
            self.base = base_policy
            self.device = base_policy.device
            self.dtype = base_policy.dtype
        def predict_action(self, obs_dict):
            return self.base.predict_action(obs_dict)
        def reset(self):
            if hasattr(self.base, "reset"):
                self.base.reset()

    policy = NativeJointPolicyAdapter(base_policy)
    log = runner.run(policy)
    print(f"\nclosed-loop rollout score (n=5): {log.get('test/mean_score')}")
    env.close()


import sys as _sys
_target = _sys.argv[1] if len(_sys.argv) > 1 else None
for name, info in TASKS.items():
    if _target is not None and name != _target:
        continue
    check_task(name, info)
