# Fig. 3 outliers

## Left panel: override error ~ 0 (< 0.1 mm), B1 success

Every point with zero override error is J7. J7 rotates about the flange axis and everything after it lies on that axis, so locking J7 leaves the end-effector POSITION unchanged and only turns the gripper about its axis (yaw). The position-only diagnosis therefore reports 0 mm by construction, while the orientation error is what makes B1 fail at the locked and severe levels. A complete predictor needs an orientation term (e.g. the error of a fingertip point offset from the flange), or J7 has to be marked and discussed separately (Fig. 3 marks it).

| B1 | level | task | joint | override mm | healthy |
|---|---|---|---|---|---|
| 0.00 | locked | Soup | J7 | 0.000 | 0.85 |
| 0.05 | locked | Bowl-Stove | J7 | 0.000 | 0.85 |
| 0.10 | severe | Bowl-Stove | J7 | 0.000 | 0.85 |
| 0.15 | severe | Soup | J7 | 0.000 | 0.85 |
| 0.20 | locked | Milk | J7 | 0.000 | 0.80 |
| 0.30 | locked | Bowl-Ramekin | J7 | 0.000 | 0.70 |
| 0.40 | moderate | Bowl-Stove | J7 | 0.000 | 0.85 |
| 0.55 | severe | Bowl-Ramekin | J7 | 0.000 | 0.70 |
| 0.65 | moderate | Milk | J7 | 0.000 | 0.80 |
| 0.70 | moderate | Bowl-Ramekin | J7 | 0.000 | 0.70 |
| 0.70 | severe | Milk | J7 | 0.000 | 0.80 |
| 0.75 | mild | Soup | J7 | 0.000 | 0.85 |
| 0.75 | moderate | Soup | J7 | 0.000 | 0.85 |
| 0.90 | mild | Bowl-Ramekin | J7 | 0.000 | 0.70 |
| 0.90 | mild | Bowl-Stove | J7 | 0.000 | 0.85 |
| 0.90 | mild | Milk | J7 | 0.000 | 0.80 |

## Right panel: residual < 3 mm but Priority IK success < 0.3

| Priority IK | level | task | joint | residual mm | healthy |
|---|---|---|---|---|---|
| 0.05 | locked | Soup | J7 | 1.51 | 0.85 |

Same cause: the residual is a position residual, and for J7 the remaining error is the gripper yaw. It is also computed on demonstration waypoints with W-IK, not on the policy's own trajectories with Priority IK.
