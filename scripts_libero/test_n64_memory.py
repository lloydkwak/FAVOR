"""
Does batch=64 (n_envs=1, n_select=64) fit in the available 10GB GPU
memory for a single predict_action call? This determines whether N=64 is
even feasible before committing a multi-hour sweep to it -- N=8 was chosen
under OOM pressure at batch=40 (n_envs=5), not because 8 candidates were
verified sufficient to capture the null-space diversity D2 measured with
N=64 offline (worth checking directly: D2 found 973/2016 sample PAIRS
within 1cm EE distance at N=64; N=8 only has C(8,2)=28 pairs total, which
may simply not contain a good candidate often enough).
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
base_policy.half()  # fp16 to roughly halve memory, testing if N=32/64 then fit

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

obs_dict = {
    "agentview_image": torch.from_numpy(to_chw(obs_src["agentview_image"][0:n_obs])[None]).to("cuda:0"),
    "robot0_eye_in_hand_image": torch.from_numpy(to_chw(obs_src["robot0_eye_in_hand_image"][0:n_obs])[None]).to("cuda:0"),
    "robot0_eef_pos": torch.from_numpy(obs_src["robot0_eef_pos"][0:n_obs][None].astype(np.float32)).to("cuda:0"),
    "robot0_eef_quat": torch.from_numpy(obs_src["robot0_eef_quat"][0:n_obs][None].astype(np.float32)).to("cuda:0"),
    "robot0_gripper_qpos": torch.from_numpy(obs_src["robot0_gripper_qpos"][0:n_obs][None].astype(np.float32)).to("cuda:0"),
}

fault_spec = {
    "q_lo": torch.full((1, 7), -2.8973).to("cuda:0"),
    "q_hi": torch.full((1, 7), 2.8973).to("cuda:0"),
}
fault_spec["q_lo"][0, 4] = 0.5
fault_spec["q_hi"][0, 4] = 0.5

for n_select in [16, 32, 48, 64]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        pol = NativeJointPolicy(base_policy, base_seed=42, mode='random_n',
                                 fault_spec=fault_spec, n_select=n_select)
        pol.reset()
        out = pol.predict_action(obs_dict)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print("n_select=%3d : OK, peak GPU memory = %.2f GB" % (n_select, peak))
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("n_select=%3d : OOM" % n_select)
        else:
            raise
