"""
Layer 1: full 21-condition kinematic upper bound, both task_dims=3
(position-only, matches fault_certificate.py's default) and task_dims=6
(position+orientation, tests the joint7 hypothesis -- see
trajectory_optimal_recovery.py's module docstring). No diffusion policy,
no GPU -- pure kinematic optimization against each task's demo_0 own
recorded joint trajectory.

Averages eps over ALL demos in each task's dataset (not just demo_0),
since a single demo's geometry could be atypical -- the mean/std across
demos is itself informative (does this joint's recoverability vary a lot
episode-to-episode, or is it consistently good/bad?).

Usage: python run_layer1_full_sweep.py
"""
import sys, os, json
sys.path.insert(0, "/workspace/docker")

import h5py
import numpy as np
import torch

from fault_kinematics import PandaKinematics
from trajectory_optimal_recovery import OptimalRecoveryComputer

JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
TASKS = {
    "alphabet_soup": "/workspace/data/robomimic/datasets/libero_alphabet_soup/ph/image_abs.hdf5",
    "milk": "/workspace/data/robomimic/datasets/libero_milk/ph/image_abs.hdf5",
    "bowl_ramekin": "/workspace/data/robomimic/datasets/libero_bowl_ramekin/ph/image_abs.hdf5",
}
OUT_DIR = "/workspace/results/layer1_optimal_recovery"
os.makedirs(OUT_DIR, exist_ok=True)

kin = PandaKinematics(device="cpu")
kin.set_base_transform(torch.tensor([-0.6, 0.0, 0.0]), torch.eye(3))

for task_name, dataset_path in TASKS.items():
    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.replace("demo_", "")))
        all_q_traj = [f[f"data/{dk}/obs/robot0_joint_pos"][:] for dk in demo_keys]

    print(f"=== {task_name}: {len(all_q_traj)} demos ===")

    for joint_name in JOINTS:
        for task_dims in (3, 6):
            fname = f"{task_name}_{joint_name}_locked_taskdims{task_dims}.json"
            out_path = os.path.join(OUT_DIR, fname)
            if os.path.exists(out_path):
                print(f"SKIP (exists): {fname}")
                continue

            computer = OptimalRecoveryComputer(kin, task_dims=task_dims)
            joint_idx = int(joint_name.replace("robot0_joint", "")) - 1

            per_demo_eps_mean = []
            per_demo_eps_max = []
            for q_traj in all_q_traj:
                q_traj_t = torch.tensor(q_traj, dtype=torch.float32)
                q_lock = q_traj_t[0, joint_idx].item()  # lock at this demo's OWN starting
                # value for that joint -- matches how fault_injector.py's real
                # fault onset works (locks at whatever q was at reset time),
                # not an arbitrary fixed value across demos
                result = computer.compute(q_traj_t, joint_name, "locked", q_lock=q_lock)
                per_demo_eps_mean.append(result["eps_mean"])
                per_demo_eps_max.append(result["eps_max"])

            summary = {
                "task": task_name, "joint": joint_name, "task_dims": task_dims,
                "n_demos": len(all_q_traj),
                "eps_mean_across_demos": float(np.mean(per_demo_eps_mean)),
                "eps_std_across_demos": float(np.std(per_demo_eps_mean)),
                "eps_max_across_demos": float(np.max(per_demo_eps_max)),
                "per_demo_eps_mean": per_demo_eps_mean,
            }
            with open(out_path, "w") as fp:
                json.dump(summary, fp, indent=2)
            print(f"  {joint_name} task_dims={task_dims}: "
                  f"eps_mean={summary['eps_mean_across_demos']:.5f} "
                  f"(+/- {summary['eps_std_across_demos']:.5f})  SAVED")

print("LAYER 1 FULL SWEEP COMPLETE")
