# M5: robomimic-layout export and diffusion_policy loading

- `scripts/m5_export.py --run out/m4_first200 --n 200 --filter success --workers 2`: 200 demonstrations
  (episodes 0-199 of the M4 run), 27,208 samples, 547 MB, images rendered offline from the saved
  MuJoCo states (5,983 s on 2 CPU cores with OSMesa; GPU EGL rendering is ~100x faster).
- Keys as `scripts_libero/convert_libero_to_robomimic.py`: `agentview_image`, `robot0_eye_in_hand_image`
  (84 x 84 x 3 uint8, HWC), `robot0_eef_pos`, `robot0_eef_quat` (xyzw), `robot0_joint_pos`, `actions`
  (T x 7 commanded joint targets), `states`, `dones`, `rewards`; per-demo attrs bowl_xy, grid_index,
  seed, n_particles, f, spill_g, success, plan parameters.
- **Image orientation**: MuJoCo's renderer already returns row 0 = top, like a RealSense frame, so no
  flip (the LIBERO flip must not be copied). `tests/test_camera.py`: the bowl projected with the
  configured OpenCV pinhole model lands 0.3 px (fovy intrinsics) / 0.5 px (fx, fy, cx, cy with an
  off-centre principal point) from the rendered bowl centroid; a vertical flip would put it 42-59 px
  away. Frame strip of one demonstration: `reports/m5_demo7_frames.png`.
- **Action is 7-D**: `dp/xarm_pour_image_joint.yaml` (shape_meta action [7], no gripper obs) with
  `dp/xarm_pour_dataset.py` (range normaliser on the joint action; see README for why not the
  upstream class).
- `tests/test_dp_dataset.py out/xarm_pour_200.hdf5`: **PASS** - 200 episodes, 27,208 steps, 25,292
  train / 516 validation samples (val_ratio 0.02), zarr cache 126 s, batch shapes images (64, 2, 3, 84, 84),
  eef_pos (64, 2, 3), eef_quat (64, 2, 4), action (64, 16, 7), normalised actions in [-0.962, 0.967],
  full epoch of 396 batches iterated without error.
- Quaternion sign continuity: largest step-to-step change of `robot0_eef_quat` over all 200 demos 0.108
  (was 1.73 with the canonical w >= 0 form).
