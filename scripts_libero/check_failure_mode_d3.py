"""
D3: What does B1 actually do under a locked joint -- for a COLLAPSED
condition (joint1, alphabet_soup, Phase 1 score 0.00) vs a MODERATE one
(joint5, alphabet_soup, Phase 1 score 0.25)?

Two questions this settles:

(a) Is the collapsed condition's failure "reasonable prediction, physically
    unreachable" (policy still predicts a sensible-looking trajectory for
    the OTHER 6 joints, it just can't compensate enough) or "prediction
    itself diverges" (the policy's non-locked-joint outputs go somewhere
    incoherent once it sees the locked joint not responding, e.g. runs to
    joint limits)? These need different fixes: the former is exactly what
    null-space steering targets; the latter would mean the policy's
    CONDITIONING is broken under this fault, which selection/compensation
    on top of a broken prediction can't repair.

(b) For the moderate condition, do episodes that SUCCEED show larger
    non-locked-joint motion (using the null space) than episodes that
    FAIL? This is the first direct behavioral evidence (rather than
    open-loop sample diversity, which D2 already checked) that null-space
    usage is what separates success from failure under this fault --
    which is the causal claim the whole design rests on.

For each condition: runs N_EP episodes closed-loop under the actual fault,
logging the full non-locked-joint trajectory and the episode outcome.
Reports (a) via per-joint range/travel relative to the arm's own physical
limits and comparison to the no-fault demo's own joint travel, and (b) via
comparing travel between successful and failed episodes.
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
from favor_fault_runner import FaultRobomimicImageRunner

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

PANDA_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "robot0_joint1"  # or robot0_joint5
N_EP = 20  # n_envs=5, so this needs multiple reset rounds to collect enough episodes

with h5py.File(DATASET, "r") as f:
    demo_q = f["data/demo_0/obs/robot0_joint_pos"][:]
demo_travel = np.abs(np.diff(demo_q, axis=0)).sum(axis=0)  # total per-joint travel in a nominal demo

payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
cls = hydra.utils.get_class(cfg._target_)
ws = cls(cfg, output_dir=None)
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = ws.ema_model if cfg.training.use_ema else ws.model
base_policy.to("cuda:0")
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/tmp/d3", dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name=CONDITION, fault_type="locked", fault_severity=None,
    n_train=0, n_test=N_EP, test_start_seed=10000, n_envs=5, max_steps=400,
    n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
    render_obs_key="agentview_image", abs_action=True,
    actuation_mode="joint", joint_kp=150,
)
policy = NativeJointPolicy(base_policy, base_seed=42, fault_spec=None)  # B1

# Instrument the run manually (rather than runner.run()) so per-episode
# joint trajectories and outcomes are both captured.
n_envs = runner.env.num_envs
n_rounds = (N_EP + n_envs - 1) // n_envs
results = []

def obs_dict_of(obs):
    return {k: torch.from_numpy(v).to("cuda:0") for k, v in obs.items()}

for round_i in range(n_rounds):
    env = runner.env
    obs = env.reset()
    policy.reset()
    joint_traj = [[] for _ in range(n_envs)]
    episode_reward = np.zeros(n_envs)
    episode_done = np.zeros(n_envs, dtype=bool)

    for step in range(400 // cfg.n_action_steps + 1):
        with torch.no_grad():
            action = policy.predict_action(obs_dict_of(obs))["action"].cpu().numpy()
        for e in range(n_envs):
            if not episode_done[e]:
                joint_traj[e].append(action[e, 0, :7].copy())
        obs, r, done, info = env.step(action)
        r_arr = np.max(np.atleast_2d(r), axis=-1) if np.ndim(r) > 1 else np.atleast_1d(r)
        episode_reward = np.maximum(episode_reward, r_arr)
        newly_done = done & ~episode_done
        for e in np.where(newly_done)[0]:
            traj = np.array(joint_traj[e])
            travel = np.abs(np.diff(traj, axis=0)).sum(axis=0) if len(traj) > 1 else np.zeros(7)
            out_of_range = np.any((traj < PANDA_Q_LO - 0.01) | (traj > PANDA_Q_HI + 0.01))
            results.append({
                "success": episode_reward[e] > 0.5,
                "travel": travel,
                "out_of_range": bool(out_of_range),
            })
        episode_done |= done
        if episode_done.all():
            break
    if len(results) >= N_EP:
        break

print("=== condition:", CONDITION, "task:", TASK, "===")
print("episodes finished:", len(results))
n_success = sum(r["success"] for r in results)
print("successes: %d/%d" % (n_success, len(results)))
print()
print("=== (a) reasonableness: predicted non-locked-joint travel vs demo's own travel, and range violations ===")
print("demo per-joint travel (rad):", np.round(demo_travel, 3))
n_oor = sum(r["out_of_range"] for r in results)
print("episodes with any joint prediction outside physical limits:", n_oor, "/", len(results))
all_travel = np.array([r["travel"] for r in results])
print("mean predicted per-joint travel across episodes:", np.round(all_travel.mean(axis=0), 3))
ratio = all_travel.mean(axis=0) / (demo_travel + 1e-6)
print("ratio to demo travel (near 1.0 = plausible magnitude, >>1 or <<0.1 = suspicious):")
print(np.round(ratio, 2))

print()
print("=== (b) success vs failure: non-locked-joint travel ===")
succ_travel = np.array([r["travel"] for r in results if r["success"]])
fail_travel = np.array([r["travel"] for r in results if not r["success"]])
if len(succ_travel) > 0:
    print("SUCCESS episodes mean travel:", np.round(succ_travel.mean(axis=0), 3))
if len(fail_travel) > 0:
    print("FAILURE episodes mean travel:", np.round(fail_travel.mean(axis=0), 3))
if len(succ_travel) > 0 and len(fail_travel) > 0:
    diff = succ_travel.mean(axis=0) - fail_travel.mean(axis=0)
    print("difference (success - failure):", np.round(diff, 3))
