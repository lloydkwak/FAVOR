"""
Convert LIBERO demonstration HDF5 files into the robomimic-style layout the
existing joint-space Diffusion Policy pipeline already consumes
(diffusion_policy/config/task/{lift,can,square}_image_joint.yaml ->
RobomimicReplayJointActionDataset -> data/robomimic/datasets/<task>/ph/image_abs.hdf5).

WHY CONVERT RATHER THAN ADAPT THE PIPELINE:
the robosuite-era pipeline is already validated end-to-end (joint-space
retraining, FaultInjector, NativeJointPolicy with deterministic seeding and
env-driven q_lock lookup). Every RoboTwin bug this project hit -- physical vs
normalized units, uncontrolled diffusion noise, q_lock approximated from
observations -- was a consequence of rewriting that machinery for a new
environment. LIBERO runs on the same robosuite/MuJoCo stack with the same
7-DoF Franka Panda, so reshaping the DATA to fit the working code is far
safer than reshaping the code to fit new data.

KEY MAPPING (verified against a real LIBERO file, not assumed):
    robomimic key              LIBERO source                     note
    -------------------------  --------------------------------  ----------------------
    robot0_eef_pos      (3)    obs/ee_pos                        identical
    robot0_eef_quat     (4)    robot_states[:, 5:9]              quaternion, norm==1
    robot0_gripper_qpos (2)    obs/gripper_states                two finger joint pos
    agentview_image            obs/agentview_rgb                 128x128x3 -> 3x84x84
    robot0_eye_in_hand_image   obs/eye_in_hand_rgb               128x128x3 -> 3x84x84
    actions             (8)    obs/joint_states(7) + actions[:,6](1)

  robot_states is [gripper(2), ee_pos(3), ee_quat(4)] -- confirmed by checking
  that robot_states[:, 2:5] == obs/ee_pos and that robot_states[:, 5:9] has
  unit norm. obs/ee_ori is axis-angle (3), NOT euler and NOT a quaternion, so
  it is deliberately unused: the quaternion already present in robot_states
  needs no conversion and therefore introduces no conversion error.

ACTION SEMANTICS (the whole point of this project):
the 8-dim action is ABSOLUTE joint positions (7) plus the original binary
gripper command (1), exactly like the robosuite lift/can/square joint configs.
LIBERO's own actions[:, :6] are OSC end-effector deltas and are discarded --
using them would put us back in the EE + IK regime whose failure modes
(controller blind to the fault; IK picking joint configurations the policy
never saw) are what motivated joint-space retraining in the first place.
"""
import argparse
import os

import h5py
import numpy as np
import cv2


IMG_SIZE = 84  # matches shape_meta [3, 84, 84] in the existing joint task configs


def resize_hwc(frames: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, size, size, 3) uint8.

    Stays in HWC on purpose. shape_meta declares [3, 84, 84] (CHW), but that
    is the shape of the FINAL tensor handed to the policy, not the layout on
    disk: RobomimicReplayJointActionDataset reads `c, h, w = shape` from
    shape_meta and then allocates its zarr array as (n_steps, h, w, c),
    encoding each frame with cv2 along the way. Handing it a CHW array makes
    the encoder fail ("Failed to encode image!"), because cv2 cannot treat a
    3xHxW buffer as an image. robomimic's own hdf5 files are HWC for the same
    reason, so matching them is what keeps this converter a drop-in.
    """
    out = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
    for t in range(frames.shape[0]):
        out[t] = cv2.resize(frames[t], (size, size), interpolation=cv2.INTER_AREA)
    return out


def convert(libero_path: str, out_path: str, verbose: bool = True) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(libero_path, "r") as src, h5py.File(out_path, "w") as dst:
        src_data = src["data"]
        dst_data = dst.create_group("data")

        demo_keys = sorted(src_data.keys(), key=lambda s: int(s.split("_")[1]))
        total_samples = 0

        for i, dk in enumerate(demo_keys):
            sd = src_data[dk]
            so = sd["obs"]

            joint = np.asarray(so["joint_states"][:], dtype=np.float32)     # (T,7)
            grip_cmd = np.asarray(sd["actions"][:, 6:7], dtype=np.float32)  # (T,1) binary -1/+1
            actions = np.concatenate([joint, grip_cmd], axis=1)             # (T,8)

            robot_states = np.asarray(sd["robot_states"][:], dtype=np.float32)
            eef_pos = np.asarray(so["ee_pos"][:], dtype=np.float32)
            eef_quat = robot_states[:, 5:9]
            grip_qpos = np.asarray(so["gripper_states"][:], dtype=np.float32)

            agentview = resize_hwc(np.asarray(so["agentview_rgb"][:]))
            eye_in_hand = resize_hwc(np.asarray(so["eye_in_hand_rgb"][:]))

            T = actions.shape[0]
            assert all(x.shape[0] == T for x in
                       (eef_pos, eef_quat, grip_qpos, agentview, eye_in_hand)), \
                f"{dk}: inconsistent episode lengths"

            g = dst_data.create_group(f"demo_{i}")
            g.attrs["num_samples"] = T
            g.create_dataset("actions", data=actions, compression="gzip")
            g.create_dataset("dones", data=np.asarray(sd["dones"][:], dtype=np.int64), compression="gzip")
            g.create_dataset("rewards", data=np.asarray(sd["rewards"][:], dtype=np.float32), compression="gzip")
            # `states` is carried over verbatim: the robomimic env_runner resets
            # episodes from it, so it must stay in LIBERO's own MuJoCo layout.
            g.create_dataset("states", data=np.asarray(sd["states"][:]), compression="gzip")

            og = g.create_group("obs")
            og.create_dataset("robot0_eef_pos", data=eef_pos, compression="gzip")
            og.create_dataset("robot0_eef_quat", data=eef_quat, compression="gzip")
            og.create_dataset("robot0_gripper_qpos", data=grip_qpos, compression="gzip")
            og.create_dataset("robot0_joint_pos", data=joint, compression="gzip")
            og.create_dataset("agentview_image", data=agentview, compression="gzip")
            og.create_dataset("robot0_eye_in_hand_image", data=eye_in_hand, compression="gzip")

            total_samples += T
            if verbose and (i % 10 == 0 or i == len(demo_keys) - 1):
                print(f"  demo_{i}: T={T}")

        dst_data.attrs["total"] = total_samples
        dst_data.attrs["num_demos"] = len(demo_keys)
        if "env_args" in src_data.attrs:
            dst_data.attrs["env_args"] = src_data.attrs["env_args"]

        if verbose:
            print(f"wrote {out_path}: {len(demo_keys)} demos, {total_samples} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(args.libero_file, args.out)


if __name__ == "__main__":
    main()
