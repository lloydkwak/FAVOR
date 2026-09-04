#!/bin/bash
# =============================================================================
# Phase 1 fault sweep (B1 baseline, no-projection policy) -- 2단계-5
#
# REVISED SCOPE (after timing measurement showed faulted episodes run to
# step_lim=400 far more often than healthy ones, making the original
# 210-condition/n=30 design take ~11 days):
#   - velocity_limited fault type DROPPED entirely for this pass (can be
#     added back as a targeted follow-up sweep later if needed).
#   - locked (no severity) + range_reduced (severity 0.25/0.5/0.75) kept,
#     preserving the "how does severity change the outcome" signal for at
#     least one fault type.
#   - n=20 episodes/condition (down from 30) -- still enough to catch large
#     effects; 95% CI roughly +/-22pp at p=0.5, adequate for Phase 1
#     screening (Phase 2 will re-verify interesting conditions at higher n
#     with B1 vs E-C-I McNemar tests).
#   - num_workers=2 per condition (measured ~1.46x throughput vs 1 worker
#     on this single 10GB GPU; 3+ workers risks OOM, untested).
#
# Grid: 3 tasks x 10 joint-arm combos (left/right x joint2-6) x 4 fault
# conditions (locked + range_reduced{0.25,0.5,0.75}) x n=20
#   = 120 conditions x 20 = 2,400 episodes total.
# joint1 excluded (see fault_injector_sapien.py docstring).
#
# One policy server per TASK (checkpoint loaded once, reused across all 40
# fault conditions for that task).
# =============================================================================
set -uo pipefail

cd ~/favor_project/RoboTwin/XPolicyLab/policy/DP

RESULTS_CSV=~/favor_project/fault_sweep_phase1_results.csv
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "task,arm,joint,fault_type,severity,success,total,rate,logfile" > "$RESULTS_CSV"
fi

N=20
NUM_WORKERS=2
TASKS=(grab_roller adjust_bottle handover_mic)
ARMS=(left right)
JOINTS=(2 3 4 5 6)

declare -A CKPT
CKPT[grab_roller]="demo_clean-grab_roller-aloha_agilex-joint-0"
CKPT[adjust_bottle]="demo_clean-adjust_bottle-aloha_agilex-joint-0"
CKPT[handover_mic]="demo_clean-handover_mic-aloha_agilex-joint-0"

for task in "${TASKS[@]}"; do
    ckpt="${CKPT[$task]}"
    policy_server_port=$(bash ../../utils/get_free_port.sh)

    echo "=== [$(date +%H:%M:%S)] Starting policy server for task=${task} (port ${policy_server_port}) ==="
    FAULT_SWEEP_HOOK=1 bash setup_eval_policy_server.sh \
        demo_clean "$task" "$ckpt" aloha_agilex joint 0 0 RoboTwin "${policy_server_port}" &
    SERVER_PID=$!
    bash ../../utils/wait_for_policy_server.sh localhost "${policy_server_port}" "${SERVER_PID}" "Policy server" 300

    for arm in "${ARMS[@]}"; do
        prefix="fl"; [[ "$arm" == "right" ]] && prefix="fr"
        for j in "${JOINTS[@]}"; do
            joint_name="${prefix}_joint${j}"

            conditions=("locked:")
            for sev in 0.25 0.5 0.75; do
                conditions+=("range_reduced:${sev}")
            done

            for cond in "${conditions[@]}"; do
                ftype="${cond%%:*}"
                sev="${cond#*:}"

                if grep -q "^${task},${arm},${joint_name},${ftype},${sev:-}," "$RESULTS_CSV" 2>/dev/null; then
                    echo "[skip, already done] ${task} ${arm} ${joint_name} ${ftype} sev=${sev:-N/A}"
                    continue
                fi

                if [[ -n "$sev" ]]; then
                    additional_info="ckpt_name=${ckpt},action_type=joint,test_num=${N},num_workers=${NUM_WORKERS},fault_arm=${arm},fault_joint=${joint_name},fault_type=${ftype},fault_severity=${sev}"
                else
                    additional_info="ckpt_name=${ckpt},action_type=joint,test_num=${N},num_workers=${NUM_WORKERS},fault_arm=${arm},fault_joint=${joint_name},fault_type=${ftype}"
                fi

                logfile="/tmp/sweep_${task}_${arm}_${joint_name}_${ftype}_${sev:-na}.log"
                FAULT_SWEEP_HOOK=1 bash setup_eval_env_client.sh \
                    demo_clean "$task" "$ckpt" aloha_agilex joint 0 0 RoboTwin \
                    "${additional_info}" "${policy_server_port}" localhost \
                    > "$logfile" 2>&1

                line=$(grep "Final batch success rate" "$logfile" | tail -1)
                succ=$(echo "$line" | grep -oP '(?<=rate: )\d+')
                total=$(echo "$line" | grep -oP '(?<=/)\d+(?= =)')
                rate=$(echo "$line" | grep -oP '(?<== )[\d.]+(?=%)')

                if [[ -z "$succ" ]]; then
                    echo "[$(date +%H:%M:%S)] WARNING: no result parsed for ${task} ${arm} ${joint_name} ${ftype} sev=${sev:-N/A} -- check $logfile"
                fi

                echo "${task},${arm},${joint_name},${ftype},${sev:-},${succ:-NA},${total:-NA},${rate:-NA},${logfile}" >> "$RESULTS_CSV"
                echo "[$(date +%H:%M:%S)] ${task} ${arm} ${joint_name} ${ftype} sev=${sev:-N/A} -> ${succ:-?}/${total:-?} = ${rate:-?}%"
            done
        done
    done

    echo "=== [$(date +%H:%M:%S)] Done with task=${task}, stopping server ==="
    kill "${SERVER_PID}" 2>/dev/null
    sleep 3
done

echo "=== Phase 1 sweep complete. Results: ${RESULTS_CSV} ==="
