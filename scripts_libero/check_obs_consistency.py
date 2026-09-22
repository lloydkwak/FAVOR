"""
(b) Observation consistency: inject demo_0 state at several timesteps,
regenerate the observation through the env adapter, compare with the
dataset's stored observation (images incl. flips, eef_quat incl. sign).
(a) Initial-scene distribution: object/robot poses after live env.reset()
vs. the 50 demos' initial states.
Usage: python check_obs_consistency.py <task>
"""
import sys
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
LOWDIM = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
IMGS = ["agentview_image", "robot0_eye_in_hand_image"]

with h5py.File(DATASET, "r") as f:
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    g = f["data/demo_0"]
    states = g["states"][:]
    dobs = {k: g["obs"][k][:] for k in LOWDIM + IMGS}
    init_states = [f[f"data/{k}/states"][0] for k in keys]

runner = FaultRobomimicImageRunner(
    output_dir=f"/workspace/results/_obs_{TASK}",
    dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint", joint_kp=150,
)
env = runner.env.env_fns[0]()
env.reset()

chain, e = [], env
while True:
    chain.append(type(e).__name__)
    if not hasattr(e, "env"):
        break
    e = e.env
raw = e
print(f"\n===== {TASK} =====\nwrapper chain: {' -> '.join(chain)}")
holder = env
while not hasattr(holder, "init_state"):
    holder = holder.env
adapter = holder.env
sim = raw.sim

def live_obs():
    if hasattr(adapter, "get_observation"):
        try:
            return adapter.get_observation(), "adapter.get_observation"
        except Exception as ex:
            print("  adapter.get_observation failed:", repr(ex))
    if hasattr(raw, "_update_observables"):
        raw._update_observables(force=True)
    return raw._get_observations(force_update=True), "raw._get_observations"

def to_hwc01(img):
    a = np.asarray(img).astype(np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.max() > 1.5:
        a = a / 255.0
    return a

# ---- (b) observation consistency ----
n = len(states)
for t in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}):
    sim.set_state_from_flattened(states[t]); sim.forward()
    o, src = live_obs()
    if t == 0:
        print(f"obs source: {src}; live keys: {sorted(o.keys())}")
    line = [f"t={t:3d}"]
    for k in LOWDIM:
        if k not in o:
            line.append(f"{k}=MISSING"); continue
        lv, dv = np.asarray(o[k]).ravel(), dobs[k][t].ravel()
        d = np.abs(lv - dv).max()
        if k == "robot0_eef_quat":
            dneg = np.abs(-lv - dv).max()
            line.append(f"quat |d|={d:.4f} |d(-q)|={dneg:.4f}")
        else:
            line.append(f"{k.replace('robot0_','')} |d|={d:.4f}")
    print("  " + "  ".join(line))
    for k in IMGS:
        if k not in o:
            print(f"    {k}: MISSING in live obs"); continue
        L, D = to_hwc01(o[k]), to_hwc01(dobs[k][t])
        if L.shape != D.shape:
            print(f"    {k}: shape mismatch live{L.shape} demo{D.shape}"); continue
        print(f"    {k}: same={np.abs(L-D).mean():.4f} vflip={np.abs(L[::-1]-D).mean():.4f} "
              f"hflip={np.abs(L[:, ::-1]-D).mean():.4f} rot180={np.abs(L[::-1, ::-1]-D).mean():.4f}")

# ---- (a) initial scene distribution ----
addrs = [sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1, 8)]

def scene():
    out = {}
    for i in range(sim.model.nbody):
        nm = sim.model.body_id2name(i)
        if nm and nm.endswith("_main") and not nm.startswith("robot") and "mount" not in nm:
            out[nm] = sim.data.body_xpos[i].copy()
    out["robot_q"] = np.array([sim.data.qpos[a] for a in addrs])
    return out

demo_sc = []
for s in init_states:
    sim.set_state_from_flattened(s); sim.forward(); demo_sc.append(scene())
live_sc = []
for _ in range(10):
    env.reset(); live_sc.append(scene())

print("\ninitial scene: demo(50) vs live reset(10)")
for k in demo_sc[0]:
    D = np.stack([s[k] for s in demo_sc]); L = np.stack([s[k] for s in live_sc])
    lo, hi = D.min(0) - 0.01, D.max(0) + 0.01
    outside = np.mean(np.any((L < lo) | (L > hi), axis=1))
    print(f"  {k:32s} demo_mean={np.round(D.mean(0),3)} demo_std={np.round(D.std(0),3)}")
    print(f"  {'':32s} live_mean={np.round(L.mean(0),3)} live_std={np.round(L.std(0),3)}  "
          f"live_outside_demo_range={outside:.0%}")
env.close()
