"""
Does the achievable minimum epsilon (over N candidates) actually keep
improving from N=8 to N=64, or does it plateau early? This is the direct
question N=8 vs N=16 vs N=64 hinges on -- D2 measured population-level
diversity (EE-close pairs spread ~80-87% as much as random pairs in joint
space), which says nothing directly about how the ACHIEVABLE MINIMUM
epsilon behaves as N grows: extreme-value statistics (the min over N iid-ish
draws) can plateau well before N reaches the full diversity sample size,
or it might not -- this has to be measured, not inferred from the
population statistic.

Draws N=64 candidates ONCE (GPU cost paid once), computes epsilon for all
64 under a locked-joint5 fault, then reports what the minimum WOULD have
been if only the first 8, 16, 32 of those 64 had been drawn -- repeated
across several random subsets at each N (not just the first-k slice) and
several timesteps, so this doesn't depend on draw order.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import h5py
import numpy as np
import torch
import dill
import hydra

from fault_kinematics import PandaKinematics
from fault_certificate import FeasibilityCertificate

TASK = "alphabet_soup"
CKPT = "/workspace/data/outputs/joint_train_libero_%s/checkpoints/latest.ckpt" % TASK
DATASET = "/workspace/data/robomimic/datasets/libero_%s/ph/image_abs.hdf5" % TASK
N_FULL = 64
JOINT_NAME = "robot0_joint5"
T0_LIST = [30, 74, 118]  # reuse a spread similar to D2's timesteps

payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
cls = hydra.utils.get_class(cfg._target_)
ws = cls(cfg, output_dir=None)
ws.load_payload(payload, exclude_keys=None, include_keys=None)
policy = ws.ema_model if cfg.training.use_ema else ws.model
policy.to("cuda:0")
policy.eval()

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

kin = PandaKinematics(device="cuda:0")
kin.set_base_transform(torch.tensor([-0.6, 0.0, 0.0], device="cuda:0"),
                        torch.eye(3, device="cuda:0"))
cert = FeasibilityCertificate(kin, beta=0.05)

# a plausible lock value: partway through this joint's own demo trajectory
with h5py.File(DATASET, "r") as f:
    demo_q5 = f["data/demo_0/obs/robot0_joint_pos"][:, 4]

for t0 in T0_LIST:
    obs_dict = {
        "agentview_image": torch.from_numpy(np.repeat(to_chw(obs_src["agentview_image"][t0:t0+n_obs])[None], N_FULL, axis=0)).to("cuda:0"),
        "robot0_eye_in_hand_image": torch.from_numpy(np.repeat(to_chw(obs_src["robot0_eye_in_hand_image"][t0:t0+n_obs])[None], N_FULL, axis=0)).to("cuda:0"),
        "robot0_eef_pos": torch.from_numpy(np.repeat(obs_src["robot0_eef_pos"][t0:t0+n_obs][None], N_FULL, axis=0).astype(np.float32)).to("cuda:0"),
        "robot0_eef_quat": torch.from_numpy(np.repeat(obs_src["robot0_eef_quat"][t0:t0+n_obs][None], N_FULL, axis=0).astype(np.float32)).to("cuda:0"),
        "robot0_gripper_qpos": torch.from_numpy(np.repeat(obs_src["robot0_gripper_qpos"][t0:t0+n_obs][None], N_FULL, axis=0).astype(np.float32)).to("cuda:0"),
    }
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    Q_full = result["action"][:, :, :7]  # (64, H, 7)

    q_lock = float(demo_q5[t0])
    eps_full, _ = cert.score(Q_full, "locked", JOINT_NAME, {"q_lock": q_lock})
    eps_full = eps_full.cpu().numpy()

    print("t0=%3d  full N=64 min epsilon = %.5f" % (t0, eps_full.min()))
    for N in [8, 16, 32]:
        mins = []
        for trial in range(20):
            idx = np.random.choice(N_FULL, size=N, replace=False)
            mins.append(eps_full[idx].min())
        mins = np.array(mins)
        print("  N=%2d : mean-of-min=%.5f  worst-of-min=%.5f  std=%.5f  (over 20 random subsets)"
              % (N, mins.mean(), mins.max(), mins.std()))
