"""
panda_fk now returns WORLD-frame positions directly (base offset baked in).
This regression test compares against sim's world-frame gripper0_grip_site
WITHOUT adding base_pos externally (that would double-count it).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import numpy as np
import torch
import robosuite
from panda_kinematics import panda_fk

env = robosuite.make(
    "Lift", robots="Panda", has_renderer=False,
    has_offscreen_renderer=False, use_camera_obs=False,
)
obs = env.reset()
sim = env.sim
np.random.seed(0)

FAILURES = 0
diffs = []
for trial in range(10):
    action = np.random.uniform(-0.3, 0.3, size=env.action_dim)
    for _ in range(5):
        obs, _, _, _ = env.step(action)

    grip_site_id = sim.model.site_name2id("gripper0_grip_site")
    grip_site_pos = sim.data.site_xpos[grip_site_id].copy()

    joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
    qpos_addrs = [sim.model.get_joint_qpos_addr(n) for n in joint_names]
    q_actual = np.array([sim.data.qpos[a] for a in qpos_addrs])
    q_t = torch.tensor(q_actual, dtype=torch.float64).unsqueeze(0)
    fk_pos, _ = panda_fk(q_t)          # world-frame directly now
    fk_world = fk_pos.squeeze(0).numpy()

    diff = np.linalg.norm(fk_world - grip_site_pos)
    diffs.append(diff)
    print(f"trial {trial}: diff={diff*1000:.3f} mm")
    if diff > 0.005:
        FAILURES += 1

env.close()
print("FAILURES", FAILURES, " max_diff_mm", max(diffs)*1000)
if FAILURES:
    sys.exit(1)
print("FK_VERIFIED_DIVERSE_CONFIGS")
