"""
Run the trained bowl_stove policy in the actual environment (closed loop,
not open-loop demo replay) starting from the demo's own initial state, and
trace what happens: does the arm reach the stove/bowl area at all? Does the
gripper close at the right time? Where does it visibly go wrong?

check_prediction_v2.py showed the policy's per-step action predictions are
close to ground truth almost everywhere (joint_err ~0.01-0.04 rad,
gripper mostly correct except a brief timing wobble around t0=38). That
rules out a systematic action-encoding bug like alphabet_soup's gripper
index bug -- so the remaining question is whether small per-step errors
are compounding into a large closed-loop drift, which open-loop metrics
on ground-truth-conditioned predictions can't reveal.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import h5py
import numpy as np
import torch
import dill
import hydra

from favor_fault_runner import FaultRobomimicImageRunner

TASK = "bowl_stove"
DATASET = "/workspace/data/robomimic/datasets/libero_%s/ph/image_abs.hdf5" % TASK
CKPT = "/workspace/data/outputs/joint_train_libero_%s/checkpoints/latest.ckpt" % TASK
SHAPE_META = {
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3]},
        "robot0_eef_quat": {"shape": [4]},
        "robot0_gripper_qpos": {"shape": [2]},
    },
    "action": {"shape": [8]},
}

payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
cls = hydra.utils.get_class(cfg._target_)
ws = cls(cfg, output_dir=None)
ws.load_payload(payload, exclude_keys=None, include_keys=None)
policy = ws.ema_model if cfg.training.use_ema else ws.model
policy.eval().to("cuda:0")

runner = FaultRobomimicImageRunner(
    output_dir="/tmp/rollout_trace", dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
    render_obs_key="agentview_image", abs_action=True, actuation_mode="joint",
)
env = runner.env.env_fns[0]()

with h5py.File(DATASET, "r") as f:
    init_state = f["data/demo_0/states"][0]
    demo_ee = f["data/demo_0/obs/robot0_eef_pos"][:]

holder = env
while not hasattr(holder, "init_state"):
    holder = holder.env
holder.init_state = init_state

obs = env.reset()
raw = env
while hasattr(raw, "env"):
    raw = raw.env

obs_dict = {k: torch.from_numpy(v[None]).to("cuda:0") for k, v in obs.items()}

best_reward = 0.0
ee_trace = []
for step in range(50):  # 50 chunks * n_action_steps ~= plenty for a 155-step demo
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    action = result["action"].cpu().numpy()  # (1, n_action_steps, 8)
    obs, r, done, info = env.step(action[0])
    best_reward = max(best_reward, float(np.max(r)))
    ee_trace.append(np.array(raw.sim.data.get_site_xpos("gripper0_grip_site")).copy()
                     if hasattr(raw.sim.data, "get_site_xpos") else None)
    obs_dict = {k: torch.from_numpy(v[None]).to("cuda:0") for k, v in obs.items()}
    if step % 5 == 0:
        gripper_qpos = raw.sim.data.qpos[raw.sim.model.get_joint_qpos_addr("gripper0_finger_joint1")] \
            if "gripper0_finger_joint1" in [raw.sim.model.joint_id2name(i) for i in range(raw.sim.model.njnt)] else None
        print("step=%3d  max_reward_so_far=%.2f" % (step, best_reward))
    if done:
        print("episode ended (done=True) at step", step)
        break

print()
print("FINAL max_reward:", best_reward, "->", "SUCCESS" if best_reward > 0.5 else "FAIL")
print("demo final EE pos:", np.round(demo_ee[-1], 3))
