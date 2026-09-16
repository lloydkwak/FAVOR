"""
Checks GPU memory for chunked N-candidate sampling at n_envs=2 (not just
n_envs=1, which test_chunked_n32.py already verified) -- actual sweep
batch per chunk is n_envs * chunk_size, not chunk_size alone. If n_envs=2
fits with chunk_size=8 (batch=16, same as the old N=8/n_envs=2 sweep's
batch), the sweep can run at half the wall-clock time of n_envs=1 for the
same N, since fewer reset rounds are needed to reach n_test=20.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import torch
import dill
import hydra

from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner

TASK = "alphabet_soup"
CKPT = "/workspace/data/outputs/joint_train_libero_%s/checkpoints/latest.ckpt" % TASK
DATASET = "/workspace/data/robomimic/datasets/libero_%s/ph/image_abs.hdf5" % TASK
SHAPE_META = {
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3]},
        "robot0_eef_quat": {"shape": [4]},
        "robot0_gripper_qpos": {"shape": [2]},
    },
    "action": {"shape": [8]},
}

payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
cls = hydra.utils.get_class(cfg._target_)
ws = cls(cfg, output_dir=None)
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to("cuda:0")
base_policy.eval()

for n_envs in [1, 2, 5]:
    for chunk_size, n_select in [(8, 32)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            runner = FaultRobomimicImageRunner(
                output_dir="/tmp/test_envs", dataset_path=DATASET, shape_meta=SHAPE_META,
                fault_joint_name="robot0_joint5", fault_type="locked", fault_severity=None,
                n_train=0, n_test=n_envs, test_start_seed=10000, n_envs=n_envs, max_steps=400,
                n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
                render_obs_key="agentview_image", abs_action=True,
                actuation_mode="joint", joint_kp=150,
            )
            policy = NativeJointPolicy(
                base_policy, base_seed=42, mode='select', env_ref=runner.env,
                fault_joint_name="robot0_joint5", fault_type="locked", fault_severity=None,
                n_select=n_select, chunk_size=chunk_size,
            )
            env = runner.env
            obs = env.reset()
            policy.reset()
            with torch.no_grad():
                result = policy.predict_action(
                    {k: torch.from_numpy(v).to("cuda:0") for k, v in obs.items()})
            peak = torch.cuda.max_memory_allocated() / 1e9
            print("n_envs=%d chunk_size=%d n_select=%d : OK, peak=%.2fGB (per-chunk batch=%d)"
                  % (n_envs, chunk_size, n_select, peak, n_envs * chunk_size))
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("n_envs=%d chunk_size=%d n_select=%d : OOM (per-chunk batch=%d)"
                      % (n_envs, chunk_size, n_select, n_envs * chunk_size))
            else:
                raise
