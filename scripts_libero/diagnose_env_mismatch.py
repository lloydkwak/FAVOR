"""
Policy-free root-cause diagnosis for bowl_stove / drawer, with
bowl_ramekin as the known-working control. State is injected directly
via sim.set_state_from_flattened (dataset states = [time, qpos, qvel]),
not via init_state + reset(), which was never verified to restore
object poses.
"""
import sys, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import h5py
import numpy as np
from favor_fault_runner import FaultRobomimicImageRunner

TASK = sys.argv[1]
DATASET = f"/workspace/data/robomimic/datasets/libero_{TASK}/ph/image_abs.hdf5"
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

print(f"\n{'='*70}\n{TASK}\n{'='*70}")

with h5py.File(DATASET, "r") as f:
    print("[1] dataset attrs:", list(f["data"].attrs.keys()))
    if "env_args" in f["data"].attrs:
        ea = json.loads(f["data"].attrs["env_args"])
        print("    env_name:", ea.get("env_name"))
        kw = ea.get("env_kwargs", {})
        for k in ("bddl_file_name", "problem_info", "controller_configs"):
            if k in kw:
                print(f"    {k}:", str(kw[k])[:300])
    states = f["data/demo_0/states"][:]
    actions = f["data/demo_0/actions"][:]
    jp = f["data/demo_0/obs/robot0_joint_pos"][:]
    dones = f["data/demo_0/dones"][:] if "dones" in f["data/demo_0"] else None
    rewards = f["data/demo_0/rewards"][:] if "rewards" in f["data/demo_0"] else None
print(f"    demo_0 len={len(actions)}  state_dim={states.shape[1]}")
if rewards is not None:
    print(f"    stored rewards: max={rewards.max()}  last={rewards[-1]}")

# [5] data format: action[t][:7] should be close to joint_pos[t+1]
d = np.abs(actions[:-1, :7] - jp[1:]).mean(axis=0)
print("[5] mean|action[t,:7] - joint_pos[t+1]| per joint:", np.round(d, 4))
print("    gripper channel unique-ish values:", np.unique(np.round(actions[:, 7], 2))[:10])

runner = FaultRobomimicImageRunner(
    output_dir=f"/workspace/results/_diag_{TASK}",
    dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint", joint_kp=150,
)
env = runner.env.env_fns[0]()
env.reset()
raw = env
while hasattr(raw, "env"):
    raw = raw.env
sim = raw.sim
print(f"    live sim: nq={sim.model.nq} nv={sim.model.nv}  "
      f"(expects state_dim = 1+nq+nv = {1+sim.model.nq+sim.model.nv})")

def body_pos(name_sub):
    out = {}
    for i in range(sim.model.nbody):
        n = sim.model.body_id2name(i)
        if n and name_sub in n:
            out[n] = np.round(sim.data.body_xpos[i], 3)
    return out

def goal_report():
    res = {}
    try:
        for g in raw.parsed_problem["goal_state"]:
            res[str(g)] = bool(raw._eval_predicate(g))
    except Exception as e:
        res["error"] = repr(e)
    return res

# [2][3] inject demo FINAL state directly
sim.set_state_from_flattened(states[-1]); sim.forward()
print("[2] final demo state injected -> _check_success():", raw._check_success())
print("    per-goal:", goal_report())
print("[3] body positions at demo final state:")
for sub in ("bowl", "plate", "stove", "cabinet", "drawer"):
    bp = body_pos(sub)
    if bp:
        print(f"    {sub}: {bp}")

# [4] inject demo START state directly, then replay demo actions
env.reset()
sim.set_state_from_flattened(states[0]); sim.forward()
best = 0.0
for t in range(len(actions)):
    _, r, done, _ = env.step(actions[t][None])
    best = max(best, float(np.max(r)))
print(f"[4] replay from exact demo start: max_reward={best:.2f}  "
      f"_check_success at end={raw._check_success()}")
print("    goal at end:", goal_report())
print("    body positions at end of replay:")
for sub in ("bowl", "plate", "stove", "cabinet", "drawer"):
    bp = body_pos(sub)
    if bp:
        print(f"    {sub}: {bp}")
env.close()
