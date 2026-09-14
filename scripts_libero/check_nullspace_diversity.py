"""
D2: Does the trained policy's multimodality include null-space diversity
-- i.e. among N samples for the same observation, are there samples with
nearly the SAME end-effector pose but DIFFERENT joint configurations?

This is the precondition the whole selection-based design (picking the
sample whose predicted execution stays closest to the intended EE path)
depends on. If the policy only varies EE-space strategy (different paths
to different places) rather than elbow configuration for the same path,
selection has nothing to select between under a locked joint -- the
compensation-only design would be needed instead.

Uses fault_kinematics.PandaKinematics (D1-verified) to get EE pose per
sample. Reports:
  1. joint-space spread (std per joint, across samples)
  2. EE-space spread (std of EE position, across samples)
  3. the actual null-space diversity metric: among sample PAIRS with EE
     positions within 1cm of each other, how much does the joint
     configuration differ? A near-zero value here means no true null-space
     diversity even if (1) is large.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401
from fault_kinematics import PandaKinematics

import h5py
import numpy as np
import torch
import dill
import hydra

TASK = "alphabet_soup"
CKPT = "/workspace/data/outputs/joint_train_libero_%s/checkpoints/latest.ckpt" % TASK
DATASET = "/workspace/data/robomimic/datasets/libero_%s/ph/image_abs.hdf5" % TASK
N_SAMPLES = 64

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
    obs_src = {
        "agentview_image": d["obs"]["agentview_image"][:],
        "robot0_eye_in_hand_image": d["obs"]["robot0_eye_in_hand_image"][:],
        "robot0_eef_pos": d["obs"]["robot0_eef_pos"][:],
        "robot0_eef_quat": d["obs"]["robot0_eef_quat"][:],
        "robot0_gripper_qpos": d["obs"]["robot0_gripper_qpos"][:],
    }

def to_chw(img):
    return (img.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

# Check diversity at a mid-episode timestep, where the arm is actively
# manipulating (more interesting than the static initial pose).
T = len(obs_src["robot0_eef_pos"])
T0_LIST = [T // 5, T // 3, T // 2, (2 * T) // 3, (4 * T) // 5]

for t0 in T0_LIST:
    print()
    print("########## t0=%d ##########" % t0)
    obs_dict_single = {
        "agentview_image": to_chw(obs_src["agentview_image"][t0:t0+n_obs])[None],
        "robot0_eye_in_hand_image": to_chw(obs_src["robot0_eye_in_hand_image"][t0:t0+n_obs])[None],
        "robot0_eef_pos": obs_src["robot0_eef_pos"][t0:t0+n_obs][None].astype(np.float32),
        "robot0_eef_quat": obs_src["robot0_eef_quat"][t0:t0+n_obs][None].astype(np.float32),
        "robot0_gripper_qpos": obs_src["robot0_gripper_qpos"][t0:t0+n_obs][None].astype(np.float32),
    }
    # replicate across the batch dim so N_SAMPLES independent denoising runs
    # happen for the SAME observation
    obs_dict = {
        k: torch.from_numpy(np.repeat(v, N_SAMPLES, axis=0)).to("cuda:0")
        for k, v in obs_dict_single.items()
    }

    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    actions = result["action"].cpu().numpy()  # (N_SAMPLES, n_action_steps, 8)

    # look at the FIRST predicted waypoint of each sample's chunk
    q_samples = actions[:, 0, :7]  # (N_SAMPLES, 7)
    print("=== 1) joint-space spread (std per joint, rad) ===")
    print(np.round(q_samples.std(axis=0), 4))

    kin = PandaKinematics(device="cpu")
    kin.set_base_transform(np.array([-0.6, 0.0, 0.0]), np.eye(3))
    ee_pos, ee_rot = kin.forward(q_samples)
    ee_pos = ee_pos.numpy()

    print()
    print("=== 2) EE-space spread (std of EE position, m) ===")
    print(np.round(ee_pos.std(axis=0), 4))
    print("mean pairwise EE distance:", round(
        np.linalg.norm(ee_pos[:, None, :] - ee_pos[None, :, :], axis=-1).mean(), 4))

    print()
    print("=== 3) null-space diversity: EE-close pairs vs ALL pairs, joint distance ===")
    ee_dist = np.linalg.norm(ee_pos[:, None, :] - ee_pos[None, :, :], axis=-1)
    q_dist = np.linalg.norm(q_samples[:, None, :] - q_samples[None, :, :], axis=-1)
    iu = np.triu_indices(N_SAMPLES, k=1)  # all unique pairs, no self-pairs, no double count
    all_q_dist = q_dist[iu]
    all_ee_dist = ee_dist[iu]
    print("ALL pairs (n=%d): mean joint dist=%.4f  mean EE dist=%.4f m"
          % (len(all_q_dist), all_q_dist.mean(), all_ee_dist.mean()))

    close_mask = all_ee_dist < 0.01
    n_close_pairs = close_mask.sum()
    print("EE-close pairs (<1cm, n=%d):" % n_close_pairs, end=" ")
    if n_close_pairs > 0:
        close_q_dists = all_q_dist[close_mask]
        print("mean joint dist=%.4f  max=%.4f" % (close_q_dists.mean(), close_q_dists.max()))
        # The real test: do EE-close pairs have joint distance comparable to
        # (or larger than) the overall population? If EE-close pairs have
        # SMALLER joint distance than the overall average, that's just the
        # ordinary correlation between EE and joint space (similar EE tends to
        # come from similar joints) -- NOT null-space diversity. Null-space
        # diversity means EE-close pairs still spread out in joint space
        # comparably to random pairs.
        ratio = close_q_dists.mean() / all_q_dist.mean()
        print("ratio (EE-close joint dist / overall joint dist): %.3f" % ratio)
        if ratio > 0.5:
            print("VERDICT: null-space diversity PRESENT -- EE-close pairs spread almost as much in")
            print("         joint space as random pairs do; elbow configuration varies independently of EE.")
        else:
            print("VERDICT: WEAK/ABSENT -- EE-close pairs are also joint-close (ratio < 0.5).")
            print("         This looks like ordinary sampling noise around one mode, not true")
            print("         null-space multimodality.")
    else:
        print("no EE-close pairs found.")
        print("VERDICT: ABSENT -- no evidence of null-space diversity at this timestep.")

    print()
    print("(caveat: overall joint std this timestep was %s rad -- if this is close to the" %
          np.round(q_samples.std(axis=0), 4))
    print(" diffusion model's inherent per-step sampling noise floor, even a ratio>0.5 verdict")
    print(" should be treated cautiously and re-checked across multiple timesteps before relying on it.)")
