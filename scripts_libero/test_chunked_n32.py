"""
Does N=32 (chunked into chunk_size=8 pieces) run without OOM, and does the
achievable minimum epsilon actually match what check_n_vs_min_epsilon.py
measured for N=32 drawn all at once (mean-of-min ~0.00075-0.00118 across
the 3 timesteps checked there)? If chunking silently produced less
diverse candidates than one true batch draw would (e.g. from a generator
seeding mistake), this would show up as a systematically worse minimum
than that reference range.
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
from fault_certificate import FeasibilityCertificate

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
    demo_q5 = f["data/demo_0/obs/robot0_joint_pos"][:, 4]

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

fault_spec = {
    "q_lo": torch.full((1, 7), -2.8973),
    "q_hi": torch.full((1, 7), 2.8973),
}

for t0 in [30, 74, 118]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q_lock = float(demo_q5[t0])
    fs = {"q_lo": fault_spec["q_lo"].clone(), "q_hi": fault_spec["q_hi"].clone()}
    fs["q_lo"][0, 4] = q_lock
    fs["q_hi"][0, 4] = q_lock

    pol = NativeJointPolicy(base_policy, base_seed=42, mode='select', fault_spec=fs,
                             fault_joint_name="robot0_joint5", fault_type="locked",
                             n_select=32, chunk_size=8,
                             base_pos=np.array([-0.6, 0.0, 0.0]), base_rot=np.eye(3))
    pol.reset()
    with torch.no_grad():
        result = pol.predict_action(obs_dict_at(t0))
    peak = torch.cuda.max_memory_allocated() / 1e9

    Q_selected = result["action_pred"][0:1, :, :7]
    eps_selected, _ = pol._cert.score(Q_selected, "locked", "robot0_joint5", {"q_lock": q_lock})
    print("t0=%3d  peak_mem=%.2fGB  selected_epsilon=%.5f" % (t0, peak, eps_selected.item()))

print()
print("reference (N=32, single-shot, from check_n_vs_min_epsilon.py):")
print("  t0=30: mean-of-min=0.00075   t0=74: mean-of-min=0.00012   t0=118: mean-of-min=0.00092")
print("(chunked select's actual selected value should land roughly in this range,")
print(" not systematically higher, which would indicate the chunking lost diversity)")
