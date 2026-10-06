# Mild range faults: where Priority IK loses to W-IK pos

Per joint, mean over the 4 tasks. dq = mean largest change of a healthy joint in the executed corrections (from the rollout logs); residual = policy-free W-IK residual (Layer1); paired = episodes only Priority IK solved : only W-IK pos solved.

| joint | B1 | W-IK pos | W-IK pose | Priority IK | paired Prio:pos | Prio dq | W-IK pos dq | residual mm |
|---|---|---|---|---|---|---|---|---|
| J1 | 0.70 | 0.76 | 0.76 | 0.81 | 8:4 | 0.107 | 0.052 | 0.4 |
| J2 | 0.30 | 0.33 | 0.34 | 0.21 | 1:10 | 0.215 | 0.078 | 13.8 |
| J3 | 0.56 | 0.80 | 0.81 | 0.81 | 10:9 | 0.034 | 0.029 | 0.3 |
| J4 | 0.60 | 0.84 | 0.77 | 0.60 | 0:19 | 0.359 | 0.059 | 6.3 |
| J5 | 0.79 | 0.83 | 0.81 | 0.83 | 5:5 | 0.122 | 0.006 | 0.0 |
| J6 | 0.80 | 0.84 | 0.70 | 0.83 | 5:6 | 0.278 | 0.025 | 0.7 |
| J7 | 0.86 | 0.88 | 0.70 | 0.81 | 1:6 | 0.590 | 0.007 | 0.1 |

Reading guide: at the mild level the fault often binds only briefly and the healthy trajectory is nearly reachable. If Priority IK's dq is much larger than W-IK pos's on the joints where it loses, the loss comes from moving healthy joints far from the policy's posture to remove a small error (W-IK pos keeps the posture via rho); the posture term that hurts on locked distal faults helps here.
