"""
Locate the fixture-region placement bug: print robosuite version,
fixture body poses, the world positions of fixture placement sites,
and the source of every placement sampler class in use.
"""
import sys, inspect
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
from favor_fault_runner import FaultRobomimicImageRunner
import robosuite

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
runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_placement_bug",
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

print("robosuite version:", robosuite.__version__, robosuite.__file__)

print("\n--- fixture body poses ---")
for nm in ("main_table", "flat_stove_1_main", "wooden_cabinet_1_main"):
    try:
        i = sim.model.body_name2id(nm)
        print(f"{nm:26s} xpos={np.round(sim.data.body_xpos[i],3)} xquat={np.round(sim.data.body_xquat[i],3)}")
    except Exception as e:
        print(nm, "not found:", e)

print("\n--- placement-related sites (world xpos) ---")
for i in range(sim.model.nsite):
    nm = sim.model.site_id2name(i)
    if nm and any(s in nm for s in ("cook_region", "top_side", "plate_region", "top_region")):
        print(f"{nm:45s} {np.round(sim.data.site_xpos[i],3)}")

print("\n--- placement initializer ---")
pi = getattr(raw, "placement_initializer", None)
print("type:", type(pi).__name__ if pi is not None else None,
      inspect.getsourcefile(type(pi)) if pi is not None else "")
seen = set()
samplers = getattr(pi, "samplers", {}) or {}
for name, s in samplers.items():
    t = type(s)
    print(f"  sampler '{name}': {t.__name__}  "
          f"reference_pos={getattr(s,'reference_pos',None)}  "
          f"objs={[getattr(o,'name',o) for o in getattr(s,'mujoco_objects',[])]}")
    if t not in seen:
        seen.add(t)
        print(f"\n===== source of {t.__name__} ({inspect.getsourcefile(t)}) =====")
        try:
            print(inspect.getsource(t))
        except Exception as e:
            print("  could not get source:", e)
env.close()
