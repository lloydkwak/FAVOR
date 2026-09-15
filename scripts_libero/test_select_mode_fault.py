"""
Verifies mode='select' under an actual fault: (1) it runs end-to-end
without error through the real env's dynamic fault_spec path
(env_ref.call('get_fault_info')/'get_base_pose'), and (2) the selected
sample's epsilon is actually <= a random sample's epsilon from the same
batch -- confirming the argmin selection is doing real work, not just
returning sample[0] by a bug.

Uses joint5 on alphabet_soup (Phase 1's "moderate" condition, score 0.25),
the condition D3 found the most behavioral evidence of null-space usage
mattering for -- the natural first place to check whether select has
something to select between.
"""
import sys
sys.path.insert(0, "/workspace/docker")
import libero_env_adapter  # noqa: F401

import numpy as np
import torch
import dill
import hydra

from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner
from fault_certificate import FeasibilityCertificate
from fault_kinematics import PandaKinematics

TASK = "alphabet_soup"
CKPT = "/workspace/data/outputs/joint_train_libero_%s/checkpoints/latest.ckpt" % TASK
DATASET = "/workspace/data/robomimic/datasets/libero_%s/ph/image_abs.hdf5" % TASK
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
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to("cuda:0")
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/tmp/test_select_fault", dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint5", fault_type="locked", fault_severity=None,
    n_train=0, n_test=1, test_start_seed=10000, n_envs=1, max_steps=400,
    n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
    render_obs_key="agentview_image", abs_action=True,
    actuation_mode="joint", joint_kp=150,
)
policy = NativeJointPolicy(
    base_policy, base_seed=42, mode='select', env_ref=runner.env,
    fault_joint_name="robot0_joint5", fault_type="locked", fault_severity=None,
    n_select=8,
)
env = runner.env
obs = env.reset()
policy.reset()

def obs_dict_of(obs):
    return {k: torch.from_numpy(v).to("cuda:0") for k, v in obs.items()}

result = policy.predict_action(obs_dict_of(obs))
print("predict_action ran successfully. action shape:", result["action"].shape)

# Manually recompute epsilon for the FIRST env's selected sample vs a
# random other sample from the same pool, using the internal certificate
# so this check doesn't rely on select's own bookkeeping being correct.
action_pred = result["action_pred"]  # (B, T, Da)
kin = policy._kin
cert = policy._cert
fault_spec = policy._dynamic_fault_spec
q_lock = fault_spec['q_lo'][0, 4].item()  # joint5 -> index 4

Q_selected = action_pred[0:1, :, :7]
eps_selected, _ = cert.score(Q_selected, "locked", "robot0_joint5", {"q_lock": q_lock})
print("selected sample epsilon:", eps_selected.item())

# draw one fresh unselected sample from the same policy/obs for comparison
# (not guaranteed worse every single time, but should be >= on average --
# check across a few draws)
worse_count = 0
n_trials = 5
for _ in range(n_trials):
    with torch.no_grad():
        nsample = base_policy.conditional_sample(
            torch.zeros(1, base_policy.horizon, base_policy.action_dim, device="cuda:0"),
            torch.zeros(1, base_policy.horizon, base_policy.action_dim, dtype=torch.bool, device="cuda:0"),
            global_cond=base_policy.obs_encoder(
                {k: v[:, :base_policy.n_obs_steps].reshape(-1, *v.shape[2:])
                 for k, v in base_policy.normalizer.normalize(obs_dict_of(obs)).items()}
            ).reshape(1, -1),
            generator=None,
        )
    Q_rand = base_policy.normalizer['action'].unnormalize(nsample[..., :base_policy.action_dim])[:, :, :7]
    eps_rand, _ = cert.score(Q_rand, "locked", "robot0_joint5", {"q_lock": q_lock})
    if eps_rand.item() >= eps_selected.item():
        worse_count += 1
    print("  random draw epsilon:", eps_rand.item())

print()
print("selected <= random in %d/%d draws" % (worse_count, n_trials))
print("PASS" if worse_count >= n_trials * 0.6 else "FAIL (or borderline -- check n_select/variance)",
      "-- selection should beat a fresh random draw most of the time")
