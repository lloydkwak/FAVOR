# M0: Menagerie ufactory_xarm7 vs official xArm7 spec

- MuJoCo python package: 3.17.0
- model: `third_party/mujoco_menagerie/ufactory_xarm7/xarm7.xml`
- integrator: mjINT_IMPLICITFAST, timestep 0.002 s
- nq 13, nv 13, nu 8

## Arm joints and actuators

| joint | MJCF range (deg) | official (deg) | diff lo / hi (deg) | damping | armature | frictionloss | actuator | kp | kv | force range (Nm) |
|---|---|---|---|---|---|---|---|---|---|---|
| J1 | -360.0 .. 360.0 | -360 .. 360 | -0.0 / +0.0 | 10 | 0.1 | 1 | general (mjDYN_NONE, bias mjBIAS_AFFINE) | 1500 | 150 | -50 .. 50 |
| J2 | -118.0 .. 120.0 | -117 .. 116 | -1.0 / +4.0 | 10 | 0.1 | 1 | general (mjDYN_NONE, bias mjBIAS_AFFINE) | 1500 | 150 | -50 .. 50 |
| J3 | -360.0 .. 360.0 | -360 .. 360 | -0.0 / +0.0 | 5 | 0.1 | 1 | general (mjDYN_NONE, bias mjBIAS_AFFINE) | 1000 | 100 | -30 .. 30 |
| J4 | -11.0 .. 225.0 | -6 .. 225 | -5.0 / +0.0 | 5 | 0.1 | 1 | general (mjDYN_NONE, bias mjBIAS_AFFINE) | 1000 | 100 | -30 .. 30 |
| J5 | -360.0 .. 360.0 | -360 .. 360 | -0.0 / +0.0 | 5 | 0.1 | 1 | general (mjDYN_NONE, bias mjBIAS_AFFINE) | 1000 | 100 | -30 .. 30 |
| J6 | -97.0 .. 180.0 | -97 .. 180 | -0.0 / -0.0 | 2 | 0.1 | 1 | general (mjDYN_NONE, bias mjBIAS_AFFINE) | 800 | 80 | -20 .. 20 |
| J7 | -360.0 .. 360.0 | -360 .. 360 | -0.0 / +0.0 | 2 | 0.1 | 1 | general (mjDYN_NONE, bias mjBIAS_AFFINE) | 800 | 80 | -20 .. 20 |

Gripper actuator: tendon 'split', ctrlrange [0.0, 255.0], gain 0.333, bias [0.0, -100.0, -10.0]

## Gravity load at the Menagerie home keyframe

| joint | gravity torque (Nm) | / force limit | steady-state error without gravity compensation (deg) |
|---|---|---|---|
| J1 | +0.00 | 0.00 | +0.000 |
| J2 | -13.51 | 0.27 | -0.516 |
| J3 | +0.13 | 0.00 | +0.007 |
| J4 | +15.56 | 0.52 | +0.891 |
| J5 | +0.07 | 0.00 | +0.004 |
| J6 | -1.53 | 0.08 | -0.110 |
| J7 | -0.00 | 0.00 | -0.000 |

## Notes

- Actuators are position servos (`general`, affine bias = PD), no velocity limit in the model; the official 180 deg/s limit is imposed by the command layer (control.max_joint_speed_deg).
- Joint ranges are overridden with the official values in the scene (robot.joint_limits_deg).
