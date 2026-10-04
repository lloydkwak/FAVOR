# J4 across fault levels

Success per task (B1 / W-IK pos / W-IK pose / Priority IK) and the policy-free kinematic errors (override = fault simply overrides the joint; residual = left after W-IK retargeting, mm).

## mild

| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | Prio dq_free | W-IK pos dq_free |
|---|---|---|---|---|---|---|---|---|
| Soup | 0.65 | 0.90 | 0.90 | 0.55 | 24.075854569673538 | 5.452756304293871 | 0.779 | 0.097 |
| Milk | 0.55 | 0.80 | 0.75 | 0.60 | 35.96043214201927 | 7.153440732508898 | 0.225 | 0.061 |
| Bowl-Ramekin | 0.75 | 0.85 | 0.75 | 0.50 | 33.288706094026566 | 7.721186615526676 | 0.240 | 0.045 |
| Bowl-Stove | 0.45 | 0.80 | 0.70 | 0.75 | 36.01095825433731 | 4.845529794692993 | 0.190 | 0.034 |

## moderate

| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | Prio dq_free | W-IK pos dq_free |
|---|---|---|---|---|---|---|---|---|
| Soup | 0.35 | 0.75 | 0.40 | 0.55 | 52.44239792227745 | 14.160627499222755 | 0.743 | 0.122 |
| Milk | 0.30 | 0.75 | 0.55 | 0.35 | 78.00813019275665 | 18.795954063534737 | 0.521 | 0.076 |
| Bowl-Ramekin | 0.05 | 0.15 | 0.00 | 0.20 | 96.49163484573364 | 28.302837163209915 | 0.375 | 0.076 |
| Bowl-Stove | 0.00 | 0.40 | 0.00 | 0.50 | 89.13667500019073 | 19.024627283215523 | 0.367 | 0.100 |

## severe

| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | Prio dq_free | W-IK pos dq_free |
|---|---|---|---|---|---|---|---|---|
| Soup | 0.05 | 0.45 | 0.15 | 0.00 | 105.39227724075317 | 32.929401844739914 | 0.708 | 0.105 |
| Milk | 0.20 | 0.65 | 0.50 | 0.45 | 156.10899031162262 | 44.873569160699844 | 0.522 | 0.056 |
| Bowl-Ramekin | 0.00 | 0.00 | 0.00 | 0.00 | 208.60466361045837 | 73.77147674560547 | 0.482 | 0.107 |
| Bowl-Stove | 0.00 | 0.00 | 0.00 | 0.00 | 156.47144615650177 | 45.728899538517 | 0.415 | 0.216 |

## locked

| task | B1 | W-IK pos | W-IK pose | Priority IK | override mm | residual mm | Prio dq_free | W-IK pos dq_free |
|---|---|---|---|---|---|---|---|---|
| Soup | 0.00 | 0.00 | 0.00 | 0.00 | 192.38047301769257 | 68.63048672676086 | 0.904 | 0.118 |
| Milk | 0.00 | 0.10 | 0.00 | 0.00 | 262.7120316028595 | 89.44639563560486 | 0.641 | 0.079 |
| Bowl-Ramekin | 0.00 | 0.00 | 0.00 | 0.00 | 340.73692560195923 | 140.6184285879135 | 0.635 | 0.180 |
| Bowl-Stove | 0.00 | 0.00 | 0.00 | 0.00 | 269.55240964889526 | 101.59890353679657 | 0.603 | 0.187 |

Reading guide: if Priority IK moves the free joints much more than W-IK pos (dq_free) while the residual is small, the drop is consistent with the missing posture term (W-IK keeps the policy's posture via rho); if the residual is large, the fault is only partly reachable and the two methods fail in different ways.
