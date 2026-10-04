# Fig. 3 outliers

## Left panel: override error ~ 0 (< 0.1 mm), B1 success

Near-zero kinematic error means the demonstrations barely move the faulty joint, so the fault costs nothing kinematically; the remaining spread should then be the policy's own success rate (compare with the healthy rate).

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

The residual is computed on demonstration waypoints with W-IK; a small residual there does not guarantee that the policy's own (shifted) trajectories stay reachable, and Priority IK's residual differs from W-IK's.
