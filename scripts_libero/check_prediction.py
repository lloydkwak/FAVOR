"""
Does the trained policy's predicted action match the demo's own recorded
action, when fed the demo's own observations?

Established so far:
  - replaying the demo's raw actions through env+FaultInjector succeeds
    (kp=150).
  - the policy, evaluated on the demo's own initial state, scores 0.
  - train_loss reached ~1e-4, which is computed in NORMALIZED action space.

If normalization is broken (wrong stats saved/restored, or a scale
mismatch between training and this construction), predicted actions could
look correct in normalized space (hence low loss) while being wrong once
un-normalized back to physical joint radians -- which would explain the
contradiction directly. This compares predicted vs. ground-truth actions in
PHYSICAL units, at the demo's own recorded observations, with no
environment execution in the loop, isolating the model+normalizer from
everything else already checked.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401 -- patches mujoco_py stub + CropRandomizer alias before anything robomimic-backed loads

import h5py
import numpy as np
import torch
import dill
import hydra

DATASET = "/workspace/data/robomimic/datasets/libero_alphabet_soup/ph/image_abs.hdf5"
CKPT = "/workspace/data/outputs/joint_train_libero_alphabet_soup/checkpoints/latest.ckpt"

payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
cls = hydra.utils.get_class(cfg._target_)
ws = cls(cfg, output_dir=None)
ws.load_payload(payload, exclude_keys=None, include_keys=None)
policy = ws.ema_model if cfg.training.use_ema else ws.model
policy.eval().to("cuda:0")

n_obs = cfg.n_obs_steps

with h5py.File(DATASET, "r") as f:
    d = f["data/demo_0"]
    actions = d["actions"][:]
    obs = {
        "agentview_image": d["obs"]["agentview_image"][:],
        "robot0_eye_in_hand_image": d["obs"]["robot0_eye_in_hand_image"][:],
        "robot0_eef_pos": d["obs"]["robot0_eef_pos"][:],
        "robot0_eef_quat": d["obs"]["robot0_eef_quat"][:],
        "robot0_gripper_qpos": d["obs"]["robot0_gripper_qpos"][:],
    }

def to_chw(img):  # (T,H,W,3) uint8 -> (T,3,H,W) float32 [0,1]
    return (img.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

t0 = 0  # first timestep, matches the very first thing the policy would see
obs_dict = {
    "agentview_image": torch.from_numpy(to_chw(obs["agentview_image"][t0:t0+n_obs])[None]).to("cuda:0"),
    "robot0_eye_in_hand_image": torch.from_numpy(to_chw(obs["robot0_eye_in_hand_image"][t0:t0+n_obs])[None]).to("cuda:0"),
    "robot0_eef_pos": torch.from_numpy(obs["robot0_eef_pos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
    "robot0_eef_quat": torch.from_numpy(obs["robot0_eef_quat"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
    "robot0_gripper_qpos": torch.from_numpy(obs["robot0_gripper_qpos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
}

with torch.no_grad():
    result = policy.predict_action(obs_dict)
pred = result["action"].cpu().numpy()[0]  # (n_action_steps, 8)

gt = actions[t0+n_obs-1 : t0+n_obs-1+pred.shape[0]]

print("predicted action[0] (joint7+gripper):", np.round(pred[0], 4))
print("ground truth action[0]              :", np.round(gt[0], 4))
print("abs diff                            :", np.round(np.abs(pred[0] - gt[0]), 4))
print()
print("predicted joint range over chunk: min=%s max=%s" % (
    np.round(pred[:, :7].min(0), 3), np.round(pred[:, :7].max(0), 3)))
print("ground truth joint range        : min=%s max=%s" % (
    np.round(gt[:, :7].min(0), 3), np.round(gt[:, :7].max(0), 3)))
print()
print("mean abs diff over chunk, joints only:", np.abs(pred[:, :7] - gt[:, :7]).mean())
