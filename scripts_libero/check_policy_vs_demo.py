"""
Final decisive check: given the SAME live observation the demo saw at
t=0, does the TRAINED POLICY predict an action close to what the demo
actually did? This directly tests generalization (not infra) -- if the
policy's own prediction already diverges significantly at t=0 from a
live observation, that's direct evidence the model failed to learn a
usable policy for this task (as opposed to some remaining infra bug).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import h5py
import numpy as np
import torch, dill, hydra
from favor_fault_runner import FaultRobomimicImageRunner  # noqa: F401 -- import side effect patches robomimic compat

import sys as _sys
TASK = _sys.argv[1] if len(_sys.argv) > 1 else "bowl_stove"
CKPT = f"/workspace/data/outputs/joint_train_libero_{TASK}/checkpoints/latest.ckpt"
DATASET = f"/workspace/data/robomimic/datasets/libero_{TASK}/ph/image_abs.hdf5"

with h5py.File(DATASET, "r") as f:
    demo_actions = f["data/demo_0/actions"][:8]  # first 8 steps (one chunk)
    demo_agentview = f["data/demo_0/obs/agentview_image"][:2]  # n_obs_steps=2
    demo_eye = f["data/demo_0/obs/robot0_eye_in_hand_image"][:2]
    demo_eef_pos = f["data/demo_0/obs/robot0_eef_pos"][:2]
    demo_eef_quat = f["data/demo_0/obs/robot0_eef_quat"][:2]
    demo_gripper = f["data/demo_0/obs/robot0_gripper_qpos"][:2]

payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir=f"/workspace/results/_scratch_policy_vs_demo")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

# build obs_dict exactly matching training format: (B=1, T=2, ...)
def to_chw(img_hwc):
    return np.transpose(img_hwc, (0, 3, 1, 2)).astype(np.float32) / 255.0

obs_dict = {
    "agentview_image": torch.from_numpy(to_chw(demo_agentview))[None].cuda(),
    "robot0_eye_in_hand_image": torch.from_numpy(to_chw(demo_eye))[None].cuda(),
    "robot0_eef_pos": torch.from_numpy(demo_eef_pos.astype(np.float32))[None].cuda(),
    "robot0_eef_quat": torch.from_numpy(demo_eef_quat.astype(np.float32))[None].cuda(),
    "robot0_gripper_qpos": torch.from_numpy(demo_gripper.astype(np.float32))[None].cuda(),
}

with torch.no_grad():
    result = base_policy.predict_action(obs_dict)
predicted = result["action"][0].cpu().numpy()  # (n_action_steps, 8)

print(f"=== {TASK}: predicted action (rows) vs demo action (given the demo's OWN first observation) ===")
print(f"predicted shape: {predicted.shape}, demo shape: {demo_actions.shape}")
n = min(len(predicted), len(demo_actions))
for i in range(n):
    diff = np.linalg.norm(predicted[i][:7] - demo_actions[i][:7])
    print(f"step {i}: predicted={np.round(predicted[i], 3)}")
    print(f"        demo     ={np.round(demo_actions[i], 3)}   joint_L2_diff={diff:.4f}")
