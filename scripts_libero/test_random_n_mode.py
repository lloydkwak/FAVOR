"""
Verifies mode='random_n': (1) reproducibility -- running twice with the
same base_seed picks the identical candidate index both times (required
for a valid paired control against select/B1 under identical seeds), and
(2) it actually varies which candidate is picked across different
episodes/calls (not silently always index 0), confirming the random draw
is real rather than degenerate.

Run with a small n_select and n_envs=1 to keep this cheap enough to run
alongside the ongoing select sweep (currently using ~2.9GB/10GB GPU
memory).
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import h5py
import numpy as np
import torch
import dill
import hydra

from native_joint_policy import NativeJointPolicy

TASK = "alphabet_soup"
CKPT = "/workspace/data/outputs/joint_train_libero_%s/checkpoints/latest.ckpt" % TASK
DATASET = "/workspace/data/robomimic/datasets/libero_%s/ph/image_abs.hdf5" % TASK

payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
cls = hydra.utils.get_class(cfg._target_)
ws = cls(cfg, output_dir=None)
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to("cuda:0")
base_policy.eval()

n_obs = cfg.n_obs_steps
with h5py.File(DATASET, "r") as f:
    d = f["data/demo_0"]
    obs_src = {
        "agentview_image": d["obs"]["agentview_image"][:],
        "robot0_eye_in_hand_image": d["obs"]["robot0_eye_in_hand_image"][:],
        "robot0_eef_pos": d["obs"]["robot0_eef_pos"][:],
        "robot0_eef_quat": d["obs"]["robot0_eef_quat"][:],
        "robot0_gripper_qpos": d["obs"]["robot0_gripper_qpos"][:],
    }

def to_chw(img):
    return (img.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

def obs_dict_at(t0):
    return {
        "agentview_image": torch.from_numpy(to_chw(obs_src["agentview_image"][t0:t0+n_obs])[None]).to("cuda:0"),
        "robot0_eye_in_hand_image": torch.from_numpy(to_chw(obs_src["robot0_eye_in_hand_image"][t0:t0+n_obs])[None]).to("cuda:0"),
        "robot0_eef_pos": torch.from_numpy(obs_src["robot0_eef_pos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
        "robot0_eef_quat": torch.from_numpy(obs_src["robot0_eef_quat"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
        "robot0_gripper_qpos": torch.from_numpy(obs_src["robot0_gripper_qpos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
    }

# NativeJointPolicy needs a real fault_spec (dict) for the random_n branch
# to trigger -- use a static one here to avoid needing a live env_ref for
# this narrow check.
fault_spec = {
    "q_lo": torch.full((1, 7), -2.8973),
    "q_hi": torch.full((1, 7), 2.8973),
}
fault_spec["q_lo"][0, 4] = 0.5  # pretend joint5 is locked at 0.5
fault_spec["q_hi"][0, 4] = 0.5

print("=== reproducibility check: same seed, same episode/call -> same choice ===")
pol_a = NativeJointPolicy(base_policy, base_seed=42, mode='random_n', fault_spec=fault_spec, n_select=4)
pol_a.reset()
out_a = pol_a.predict_action(obs_dict_at(0))["action"]

pol_b = NativeJointPolicy(base_policy, base_seed=42, mode='random_n', fault_spec=fault_spec, n_select=4)
pol_b.reset()
out_b = pol_b.predict_action(obs_dict_at(0))["action"]

print("PASS" if torch.allclose(out_a, out_b) else "FAIL", "-- identical seed must give identical random_n choice")

print()
print("=== variation check: different episode indices should (usually) pick different candidates ===")
choices = []
for ep in range(6):
    pol = NativeJointPolicy(base_policy, base_seed=42, mode='random_n', fault_spec=fault_spec, n_select=4)
    for _ in range(ep):
        pol.reset()  # advance episode_idx without predicting, to vary the seed cheaply
    pol.reset()
    out = pol.predict_action(obs_dict_at(0))["action"]
    choices.append(out[0, 0, 0].item())  # a cheap fingerprint of which candidate was chosen

n_unique = len(set(np.round(choices, 5)))
print("fingerprints across 6 episodes:", [round(c, 4) for c in choices])
print("PASS" if n_unique > 1 else "FAIL", "-- should not always pick the same candidate (n_unique=%d/6)" % n_unique)
