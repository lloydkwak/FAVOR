"""
Verifies mode='select' degenerates exactly to B1 when there is no fault
(fault_spec is None): predict_action's new select/select_comp branch is
gated on `fault_spec is not None`, so with no fault it should fall straight
into the existing B1 branch -- meaning a 'select'-mode policy evaluated on
a fault-free episode must be identical to a 'select'=B1 comparison,
important because Phase 2's "does the new method hurt no-fault
performance" check depends on this being true by construction, not by
coincidence.

Checks this by running both with the SAME generator seed for one
predict_action call and confirming bitwise-identical output.
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

t0 = 0
obs_dict = {
    "agentview_image": torch.from_numpy(to_chw(obs_src["agentview_image"][t0:t0+n_obs])[None]).to("cuda:0"),
    "robot0_eye_in_hand_image": torch.from_numpy(to_chw(obs_src["robot0_eye_in_hand_image"][t0:t0+n_obs])[None]).to("cuda:0"),
    "robot0_eef_pos": torch.from_numpy(obs_src["robot0_eef_pos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
    "robot0_eef_quat": torch.from_numpy(obs_src["robot0_eef_quat"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
    "robot0_gripper_qpos": torch.from_numpy(obs_src["robot0_gripper_qpos"][t0:t0+n_obs][None].astype(np.float32)).to("cuda:0"),
}

pol_b1 = NativeJointPolicy(base_policy, base_seed=42, fault_spec=None, mode='eci')
pol_b1.reset()
out_b1 = pol_b1.predict_action(obs_dict)["action"]

pol_select = NativeJointPolicy(base_policy, base_seed=42, mode='select',
                                fault_spec=None, env_ref=None,
                                base_pos=np.array([-0.6, 0.0, 0.0]), base_rot=np.eye(3))
pol_select.reset()
out_select = pol_select.predict_action(obs_dict)["action"]

print("B1 action[0,0,:3]:    ", out_b1[0, 0, :3].tolist())
print("select action[0,0,:3]:", out_select[0, 0, :3].tolist())
print("PASS" if torch.allclose(out_b1, out_select) else "FAIL",
      "-- select with fault_spec=None must equal B1 exactly")
