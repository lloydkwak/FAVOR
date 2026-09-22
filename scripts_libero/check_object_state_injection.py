"""
Last diagnostic round: does env.reset() with init_state set actually
restore the BOWL's own position, or does reset() re-randomize object
placement from the bddl's init regions (ignoring init_state for objects)?

If reset() ignores init_state for objects: the previous
check_replay_endpoint.py test was invalid (bowl position was random each
step, not the demo's true position) -- meaning the predicate has NOT
actually been shown to fail; the methodology was flawed.

If reset() correctly restores object position from init_state: the
predicate genuinely does not fire even with everything else correct,
which would be a real, confirmed task-specific bug.

Also checks the sim's raw qpos length/structure to see how object state
is laid out, since this env wraps bddl-based object placement.
"""
import sys
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
    demo_states = f["data/demo_0/states"][:]
    demo_len = len(demo_states)

print(f"demo_0 length: {demo_len}, state vector dim: {demo_states.shape[1]}")

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_obj_state_check",
    dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint", joint_kp=150,
)
env = runner.env.env_fns[0]()
holder = env
while not hasattr(holder, "init_state"):
    holder = holder.env
raw = env
while hasattr(raw, "env"):
    raw = raw.env

# Inject the demo's OWN FINAL state (must correspond to success) and see
# what qpos the sim actually ends up with, vs what the demo saved.
final_state = demo_states[-1]
holder.init_state = final_state
env.reset()

sim_qpos_after_reset = np.array(raw.sim.data.qpos[:])
print(f"\nfull sim qpos dim after reset with final_state injected: {sim_qpos_after_reset.shape}")
print(f"stored demo final_state dim: {final_state.shape}")
print(f"do they match in length? {sim_qpos_after_reset.shape[0] == final_state.shape[0]}")

# Direct comparison: how far is the injected sim qpos from the literal
# stored final_state vector (same indexing, since robosuite's state IS
# essentially the full mjstate/qpos+qvel concatenation for this env type)
if sim_qpos_after_reset.shape[0] == final_state.shape[0]:
    diff = np.abs(sim_qpos_after_reset - final_state[: len(sim_qpos_after_reset)])
    print(f"max abs diff between injected qpos and stored final_state: {diff.max():.6f}")
    print(f"mean abs diff: {diff.mean():.6f}")
    if diff.max() < 1e-3:
        print(">>> State injection appears to work correctly (qpos matches stored state).")
    else:
        print(">>> MISMATCH: reset() is NOT faithfully restoring the stored state "
              "(likely re-randomizing object placement) -- prior predicate-fail result is INVALID.")
else:
    print(">>> dim mismatch -- state vector layout differs from raw sim.qpos, "
          "cannot directly compare this way; need a different state-setting API.")

# Try the predicate check right after this direct injection
if hasattr(raw, "_check_success"):
    r = raw._check_success()
    print(f"\n_check_success() right after injecting demo's OWN final state: {r}")
else:
    print("\nraw has no _check_success attribute")
    print("attrs containing 'succ'/'check'/'goal':",
          [a for a in dir(raw) if any(k in a.lower() for k in ("succ", "check", "goal"))])

env.close()
