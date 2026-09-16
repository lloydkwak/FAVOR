"""
Are the 4 chunks (chunk_size=8, N=32) actually producing DIFFERENT samples,
or are chunks 2-4 silently duplicating chunk 1 (e.g. because
_next_generator's seed formula doesn't actually vary within one
predict_action call, or because conditional_sample ignores/mishandles a
freshly-seeded CPU generator for a CUDA op)?

Compares pairwise distances WITHIN chunk 1 (8 samples) against distances
ACROSS chunks (chunk 1 vs chunk 2, etc). If chunking is broken, cross-chunk
distances would be near zero while within-chunk distances are normal.
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

t0 = 74
obs_dict = {
    "agentview_image": torch.from_numpy(to_chw(obs_src["agentview_image"][t0:t0+n_obs])[None]).to("cuda:0"),
    "robot0_eye_in_hand_image": torch.from_numpy(to_chw(obs_src["robot0_eye_in_hand_image"][t0:t0+n_obs])[None]).to("cuda:0"),
    "robot0_eef_pos": torch.from_numpy(obs_src["robot0_eef_pos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
    "robot0_eef_quat": torch.from_numpy(obs_src["robot0_eef_quat"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
    "robot0_gripper_qpos": torch.from_numpy(obs_src["robot0_gripper_qpos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
}

fault_spec = {"q_lo": torch.full((1, 7), -2.8973), "q_hi": torch.full((1, 7), 2.8973)}
pol = NativeJointPolicy(base_policy, base_seed=42, mode='select', fault_spec=fault_spec,
                         fault_joint_name="robot0_joint5", fault_type="locked",
                         n_select=32, chunk_size=8,
                         base_pos=np.array([-0.6, 0.0, 0.0]), base_rot=np.eye(3))
pol.reset()

with torch.no_grad():
    nobs = base_policy.normalizer.normalize(obs_dict)
    this_nobs = {k: v[:, :base_policy.n_obs_steps].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
    global_cond = base_policy.obs_encoder(this_nobs).reshape(1, -1)
    T = base_policy.horizon
    Da = base_policy.action_dim
    cond_data = torch.zeros(1, T, Da, device="cuda:0")
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

    samples = pol._draw_n_candidates_chunked(cond_data, cond_mask, global_cond, 32)  # (1, 32, T, Da)

Q = samples[0, :, :, :7].cpu().numpy()  # (32, T, 7), N=32 samples of 7-joint trajectories
Q_flat = Q.reshape(32, -1)

within_c1 = []
for i in range(8):
    for j in range(i+1, 8):
        within_c1.append(np.linalg.norm(Q_flat[i] - Q_flat[j]))

cross_c1_c2 = []
for i in range(8):
    for j in range(8, 16):
        cross_c1_c2.append(np.linalg.norm(Q_flat[i] - Q_flat[j]))

cross_c1_c3 = [np.linalg.norm(Q_flat[i] - Q_flat[j]) for i in range(8) for j in range(16, 24)]
cross_c1_c4 = [np.linalg.norm(Q_flat[i] - Q_flat[j]) for i in range(8) for j in range(24, 32)]

print("within chunk1 (8 samples, same generator instance):      mean_dist=%.4f" % np.mean(within_c1))
print("chunk1 vs chunk2 (should be similar if chunking is OK):  mean_dist=%.4f" % np.mean(cross_c1_c2))
print("chunk1 vs chunk3:                                        mean_dist=%.4f" % np.mean(cross_c1_c3))
print("chunk1 vs chunk4:                                        mean_dist=%.4f" % np.mean(cross_c1_c4))
print()
print("if cross-chunk << within-chunk, chunks are duplicating each other (BROKEN)")
print("if cross-chunk ~= within-chunk, chunking is producing real independent diversity (OK)")
