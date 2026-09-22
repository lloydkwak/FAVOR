"""
Replay demo actions from each demo's EXACT start state (direct
set_state_from_flattened), at a given joint_kp, over the first 5 demos.
Reports success rate, joint tracking error, and (for drawer) how far the
middle drawer opened vs. the demo's own final opening.
Usage: python replay_kp_sweep.py <kp> <task>
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import h5py
import numpy as np
from favor_fault_runner import FaultRobomimicImageRunner

KP = int(sys.argv[1])
TASK = sys.argv[2]
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
N_DEMOS = 5

runner = FaultRobomimicImageRunner(
    output_dir=f"/workspace/results/_kp_{TASK}_{KP}",
    dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint", joint_kp=KP,
)
env = runner.env.env_fns[0]()
env.reset()
raw = env
while hasattr(raw, "env"):
    raw = raw.env
sim = raw.sim
addrs = [sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1, 8)]

def drawer_opening():
    try:
        mid = sim.data.body_xpos[sim.model.body_name2id("wooden_cabinet_1_cabinet_middle")]
        base = sim.data.body_xpos[sim.model.body_name2id("wooden_cabinet_1_base")]
        return float(abs(mid[1] - base[1]))
    except Exception:
        return float("nan")

n_succ = 0
with h5py.File(DATASET, "r") as f:
    for d in range(N_DEMOS):
        g = f[f"data/demo_{d}"]
        states, actions = g["states"][:], g["actions"][:]
        jp = g["obs/robot0_joint_pos"][:]

        sim.set_state_from_flattened(states[-1]); sim.forward()
        demo_open = drawer_opening()

        env.reset()
        sim.set_state_from_flattened(states[0]); sim.forward()
        best, errs = 0.0, []
        for t in range(len(actions)):
            _, r, _, _ = env.step(actions[t][None])
            best = max(best, float(np.max(r)))
            if t + 1 < len(jp):
                q = np.array([sim.data.qpos[a] for a in addrs])
                errs.append(np.linalg.norm(q - jp[t + 1]))
        ok = best > 0.5
        n_succ += ok
        print(f"kp={KP} {TASK} demo_{d}: success={ok}  "
              f"track_err mean={np.mean(errs):.4f} max={np.max(errs):.4f}  "
              f"drawer_open replay={drawer_opening():.3f} demo={demo_open:.3f}")

print(f"==> kp={KP} {TASK}: replay success {n_succ}/{N_DEMOS}")
env.close()
