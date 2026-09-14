"""
check_prediction.py, generalized to any task, run for bowl_stove alone
(one process, per the deadlock pattern noted elsewhere).

bowl_stove and drawer trained cleanly (loss converged normally) and their
success predicates fire correctly on demo replay, yet both score 0% at
every rollout -- the same shape of contradiction that turned out to be a
gripper-channel bug for alphabet_soup. This compares predicted vs.
ground-truth actions channel-by-channel at several points in a demo, not
just the first timestep, to catch a problem that might only appear later
in the episode (e.g. right before the grasp/release that gripper timing
depends on).
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import h5py
import numpy as np
import torch
import dill
import hydra

TASK = "bowl_stove"
DATASET = "/workspace/data/robomimic/datasets/libero_%s/ph/image_abs.hdf5" % TASK
CKPT = "/workspace/data/outputs/joint_train_libero_%s/checkpoints/latest.ckpt" % TASK

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

def to_chw(img):
    return (img.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

T = len(actions)
# sample a few points across the episode, including near the end where
# a grasp/release (gripper timing) is most likely to matter
sample_starts = sorted(set([0, T // 4, T // 2, (3 * T) // 4, max(0, T - n_obs - 8)]))

for t0 in sample_starts:
    obs_dict = {
        "agentview_image": torch.from_numpy(to_chw(obs["agentview_image"][t0:t0+n_obs])[None]).to("cuda:0"),
        "robot0_eye_in_hand_image": torch.from_numpy(to_chw(obs["robot0_eye_in_hand_image"][t0:t0+n_obs])[None]).to("cuda:0"),
        "robot0_eef_pos": torch.from_numpy(obs["robot0_eef_pos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
        "robot0_eef_quat": torch.from_numpy(obs["robot0_eef_quat"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
        "robot0_gripper_qpos": torch.from_numpy(obs["robot0_gripper_qpos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
    }
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    pred = result["action"].cpu().numpy()[0]
    gt = actions[t0+n_obs-1 : t0+n_obs-1+pred.shape[0]]
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]

    joint_err = np.abs(pred[:, :7] - gt[:, :7]).mean()
    grip_err = np.abs(pred[:, 7] - gt[:, 7]).mean()
    print("t0=%3d  joint_err=%.4f  gripper_err=%.4f  pred_grip[0]=%.3f  gt_grip[0]=%.3f"
          % (t0, joint_err, grip_err, pred[0, 7], gt[0, 7]))
