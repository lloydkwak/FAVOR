# J4 across fault levels

Success per task (B1 / W-IK pos / W-IK pose / Priority IK) and the policy-free kinematic errors (override = fault simply overrides the joint; residual = left after W-IK retargeting, mm).

## mild

| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | Prio dq_free | W-IK pos dq_free |
|---|---|---|---|---|---|---|---|---|
| Soup | 0.65 | 0.90 | 0.90 | 0.55 | 24.1 | 5.5 | 0.779 | 0.097 |
| Milk | 0.55 | 0.80 | 0.75 | 0.60 | 36.0 | 7.2 | 0.225 | 0.061 |
| Bowl-Ramekin | 0.75 | 0.85 | 0.75 | 0.50 | 33.3 | 7.7 | 0.240 | 0.045 |
| Bowl-Stove | 0.45 | 0.80 | 0.70 | 0.75 | 36.0 | 4.8 | 0.190 | 0.034 |

## moderate

| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | Prio dq_free | W-IK pos dq_free |
|---|---|---|---|---|---|---|---|---|
| Soup | 0.35 | 0.75 | 0.40 | 0.55 | 52.4 | 14.2 | 0.743 | 0.122 |
| Milk | 0.30 | 0.75 | 0.55 | 0.35 | 78.0 | 18.8 | 0.521 | 0.076 |
| Bowl-Ramekin | 0.05 | 0.15 | 0.00 | 0.20 | 96.5 | 28.3 | 0.375 | 0.076 |
| Bowl-Stove | 0.00 | 0.40 | 0.00 | 0.50 | 89.1 | 19.0 | 0.367 | 0.100 |

## severe

| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | Prio dq_free | W-IK pos dq_free |
|---|---|---|---|---|---|---|---|---|
| Soup | 0.05 | 0.45 | 0.15 | 0.00 | 105.4 | 32.9 | 0.708 | 0.105 |
| Milk | 0.20 | 0.65 | 0.50 | 0.45 | 156.1 | 44.9 | 0.522 | 0.056 |
| Bowl-Ramekin | 0.00 | 0.00 | 0.00 | 0.00 | 208.6 | 73.8 | 0.482 | 0.107 |
| Bowl-Stove | 0.00 | 0.00 | 0.00 | 0.00 | 156.5 | 45.7 | 0.415 | 0.216 |

## locked

| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | Prio dq_free | W-IK pos dq_free |
|---|---|---|---|---|---|---|---|---|
| Soup | 0.00 | 0.00 | 0.00 | 0.00 | 192.4 | 68.6 | 0.904 | 0.118 |
| Milk | 0.00 | 0.10 | 0.00 | 0.00 | 262.7 | 89.4 | 0.641 | 0.079 |
| Bowl-Ramekin | 0.00 | 0.00 | 0.00 | 0.00 | 340.7 | 140.6 | 0.635 | 0.180 |
| Bowl-Stove | 0.00 | 0.00 | 0.00 | 0.00 | 269.6 | 101.6 | 0.603 | 0.187 |

Reading guide: if Priority IK moves the free joints much more than W-IK pos (dq_free) while the residual is small, the drop is consistent with the missing posture term (W-IK keeps the policy's posture via rho); if the residual is large, the fault is only partly reachable and the two methods fail in different ways.
