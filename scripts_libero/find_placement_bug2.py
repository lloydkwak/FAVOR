"""
(1) First reset vs subsequent resets: is bowl placement correct on the
    first reset and wrong afterwards (soft-reset hypothesis)?
(2) Print the env's hard_reset setting and all placement-related attrs.
(3) Print source of the code that places objects on fixture regions.
"""
import sys, inspect
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
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
runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_placement_bug2",
    dataset_path=DATASET, shape_meta=SHAPE_META,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=1, n_test=0, n_envs=1, train_start_idx=0, max_steps=400,
    n_obs_steps=2, n_action_steps=1, render_obs_key="agentview_image",
    abs_action=True, actuation_mode="joint", joint_kp=150,
)
env = runner.env.env_fns[0]()
raw = env
while hasattr(raw, "env"):
    raw = raw.env

def report(tag):
    sim = raw.sim
    def bp(n):
        return np.round(sim.data.body_xpos[sim.model.body_name2id(n)], 3)
    def sp(n):
        return np.round(sim.data.site_xpos[sim.model.site_name2id(n)], 3)
    print(f"[{tag}] bowl_1={bp('akita_black_bowl_1_main')}  cook_site={sp('flat_stove_1_cook_region')}  "
          f"| bowl_2={bp('akita_black_bowl_2_main')}  top_site={sp('wooden_cabinet_1_top_side')}")

print("hard_reset:", getattr(raw, "hard_reset", "N/A"))
print("placement-related attrs:")
for a in dir(raw):
    if "placement" in a.lower() or "initializer" in a.lower():
        v = getattr(raw, a, None)
        if callable(v) and not hasattr(v, "samplers"):
            continue
        print(f"  {a}: {type(v).__name__}")
        for nm, s in (getattr(v, "samplers", {}) or {}).items():
            print(f"      sampler '{nm}': {type(s).__name__} reference_pos={getattr(s,'reference_pos',None)}")

for k in range(3):
    env.reset()
    report(f"reset #{k+1}")

print("\n===== source: _reset_internal =====")
try:
    print(inspect.getsource(type(raw)._reset_internal))
except Exception as e:
    print("n/a:", e)
for m in ("_setup_placement_initializer", "_load_fixtures_in_arena", "_load_objects_in_arena"):
    for cls in type(raw).__mro__:
        if m in cls.__dict__:
            print(f"\n===== source: {cls.__name__}.{m} ({inspect.getsourcefile(cls)}) =====")
            print(inspect.getsource(cls.__dict__[m]))
            break
env.close()
