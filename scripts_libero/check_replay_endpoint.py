"""
Follow-up to check_action_replay.py's FAIL result: WHERE does the live
action replay diverge from the demo? Two sub-hypotheses:

  (A) Trajectory tracking fails badly (final joint/eef state far from
      demo's own final state) -> physics/controller problem.
  (B) Trajectory tracking is fine (final state close to demo's final
      state) but reward/predicate still doesn't fire -> predicate
      evaluation problem specific to this task/bddl (e.g. checking the
      wrong object name, or a site/geom that doesn't match this bddl's
      actual object naming).

Also runs GROUND-TRUTH STATE replay (env.set_state at each demo timestep,
not action-driven) as an independent check of the predicate alone,
decoupled from physics/controller entirely -- this isolates predicate
correctness from tracking accuracy.
"""
import sys, os
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import h5py
import numpy as np

from favor_fault_runner import FaultRobomimicImageRunner

DATASET = "/workspace/data/robomimic/datasets/libero_bowl_stove/ph/image_abs.hdf5"
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

with h5py.File(DATASET, "r") as f:
    demo_actions = f["data/demo_0/actions"][:]
    demo_states = f["data/demo_0/states"][:]
    demo_q = f["data/demo_0/obs/robot0_joint_pos"][:]
    demo_eef = f["data/demo_0/obs/robot0_eef_pos"][:]
demo_len = len(demo_actions)
print(f"demo_0 length: {demo_len} steps")
print(f"demo final joint_pos: {np.round(demo_q[-1], 4)}")
print(f"demo final eef_pos:   {np.round(demo_eef[-1], 4)}")

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_replay_endpoint",
    dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint", joint_kp=150,
)

print("\n--- (1) GROUND-TRUTH STATE replay (predicate isolation) ---")
env = runner.env.env_fns[0]()
holder = env
while not hasattr(holder, "init_state"):
    holder = holder.env
raw = env
while hasattr(raw, "env"):
    raw = raw.env

best_reward_state = 0.0
env.reset()
for t in range(demo_len):
    holder.init_state = demo_states[t]
    env.reset()  # forces state injection via init_state on this wrapper
    if hasattr(raw, "_check_success"):
        r = raw._check_success()
    else:
        if t == 0:
            print("  NOTE: 'raw' has no _check_success; available attrs matching 'succ' or 'check':",
                  [a for a in dir(raw) if "succ" in a.lower() or "check" in a.lower()])
        r = None
    if r:
        best_reward_state = 1.0
        print(f"  predicate fired TRUE at demo step {t}")
        break
print(f"state-replay predicate result: {'SUCCESS' if best_reward_state > 0.5 else 'NEVER FIRED (checked all steps)'}")
env.close()

print("\n--- (2) ACTION replay, tracking accuracy at final step ---")
env = runner.env.env_fns[0]()
holder = env
while not hasattr(holder, "init_state"):
    holder = holder.env
with h5py.File(DATASET, "r") as f:
    holder.init_state = f["data/demo_0/states"][0]
env.reset()
raw = env
while hasattr(raw, "env"):
    raw = raw.env
addrs = [raw.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1, 8)]

best_reward = 0.0
for t in range(demo_len):
    action = demo_actions[t][None]
    _, r, done, _ = env.step(action)
    best_reward = max(best_reward, float(np.max(r)))

q_final = np.array([raw.sim.data.qpos[a] for a in addrs])
print(f"achieved final joint_pos: {np.round(q_final, 4)}")
print(f"|achieved - demo_final| joint diff L2: {np.linalg.norm(q_final - demo_q[-1]):.4f}")
print(f"final reward achieved: {best_reward:.2f}")
env.close()
