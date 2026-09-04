#!/bin/bash
# =============================================================================
# Phase 2 (2단계-6): B1 vs posthoc vs eci, paired comparison on 8 anchor
# conditions selected OBJECTIVELY from Phase 1 by observed B1 success-rate
# tier (collapse <=10%, moderate 11-50%, resilient >50%), one joint per
# tier per task -- NOT hand-picked for a favorable outcome. handover_mic
# has no resilient-tier joint at all (max observed was 25%), so it
# contributes only 2 anchors instead of 3; this asymmetry is itself a
# task-level finding worth reporting, not an error.
#
# fault_type=locked only for this pass (cleanest identity-projection
# story on the policy side, no severity confound). range_reduced follow-up
# is a natural Phase 2b once this is validated.
#
# For McNemar pairing: env-side fault (FaultInjector via fault_sweep_hook,
# same mechanism as Phase 1) is IDENTICAL across b1/posthoc/eci for a given
# condition. Only the POLICY SERVER's sampling_mode differs (b1=default/
# untouched, posthoc/eci=policy_fault_hook active). Because expert_check
# (which determines which seeds get skipped) never depends on sampling_mode
# or env-side fault, the set of seeds actually evaluated is naturally
# identical across the three runs for the same condition -- pairing is done
# by matching on `seed`, not on episode order, when analyzing results.
#
# PER-SEED results (not just aggregate) are captured here, required for
# McNemar -- this is the key difference from Phase 1's driver.
# =============================================================================
set -uo pipefail

cd ~/favor_project/RoboTwin/XPolicyLab/policy/DP

RESULTS_CSV=~/favor_project/fault_sweep_phase2_results.csv
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "task,arm,joint,fault_type,sampling_mode,seed,worker,success" > "$RESULTS_CSV"
fi

SUMMARY_CSV=~/favor_project/fault_sweep_phase2_summary.csv
if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "task,arm,joint,fault_type,sampling_mode,success,total,rate" > "$SUMMARY_CSV"
fi

N=50
NUM_WORKERS=1
MODES=(b1 posthoc eci)

declare -A CKPT
CKPT[grab_roller]="demo_clean-grab_roller-aloha_agilex-joint-0"
CKPT[adjust_bottle]="demo_clean-adjust_bottle-aloha_agilex-joint-0"
CKPT[handover_mic]="demo_clean-handover_mic-aloha_agilex-joint-0"

# task, arm, joint  (objectively selected anchors, see header comment)
ANCHORS=(
  "grab_roller left fl_joint4"
  "grab_roller right fr_joint3"
  "grab_roller left fl_joint5"
  "adjust_bottle left fl_joint3"
  "adjust_bottle left fl_joint6"
  "adjust_bottle right fr_joint5"
  "handover_mic right fr_joint3"
  "handover_mic right fr_joint6"
)

for anchor in "${ANCHORS[@]}"; do
    read -r task arm joint <<< "$anchor"
    ckpt="${CKPT[$task]}"

    for mode in "${MODES[@]}"; do
        cond_key="${task},${arm},${joint},locked,${mode}"
        if grep -q "^${cond_key}," "$SUMMARY_CSV" 2>/dev/null; then
            echo "[skip, already done] ${task} ${arm} ${joint} mode=${mode}"
            continue
        fi

        echo "=== [$(date +%H:%M:%S)] ${task} ${arm} ${joint} locked mode=${mode} ==="

        policy_server_port=$(bash ../../utils/get_free_port.sh)

        # Server-side overrides: sampling_mode always set; fault_* only
        # matters for posthoc/eci (policy_fault_hook.py is a no-op under
        # mode=b1 regardless of these being present).
        server_overrides="bench_name=demo_clean task_name=${task} ckpt_name=${ckpt} env_cfg_type=aloha_agilex seed=0 policy_name=DP action_type=joint sampling_mode=${mode} fault_arm=${arm} fault_joint=${joint} fault_type=locked"

        python ../../setup_policy_server.py \
            --config_path deploy.yml \
            --overrides ${server_overrides} port=${policy_server_port} host=localhost \
            > /tmp/phase2_server_${task}_${arm}_${joint}_${mode}.log 2>&1 &
        SERVER_PID=$!

        bash ../../utils/wait_for_policy_server.sh localhost "${policy_server_port}" "${SERVER_PID}" "Policy server" 300

        # Env-side: SAME fault regardless of mode (only the policy's
        # sampling changes across b1/posthoc/eci).
        additional_info="ckpt_name=${ckpt},action_type=joint,test_num=${N},num_workers=${NUM_WORKERS},fault_arm=${arm},fault_joint=${joint},fault_type=locked"

        logfile="/tmp/phase2_${task}_${arm}_${joint}_${mode}.log"
        FAULT_SWEEP_HOOK=1 bash ../../policy/DP/setup_eval_env_client.sh \
            demo_clean "$task" "$ckpt" aloha_agilex joint 0 0 RoboTwin \
            "${additional_info}" "${policy_server_port}" localhost \
            > "$logfile" 2>&1

        kill "${SERVER_PID}" 2>/dev/null
        sleep 2

        # extract per-seed lines: ANSI escape codes and \r progress-bar
        # characters can obscure these lines from naive grep (confirmed
        # during Phase 2 pilot debugging) -- strip both before matching.
        python3 -c "
import re
task = '${task}'
with open('${logfile}', 'rb') as f:
    raw = f.read().decode('utf-8', errors='replace')
clean = re.sub(r'\x1b\[[0-9;]*m', '', raw).replace('\r', '\n')
pattern = re.compile(task + r' \| episode=\d+ \| worker=(\d+) \| seed=(\d+) \| success=(True|False)')
with open('${RESULTS_CSV}', 'a') as out:
    for m in pattern.finditer(clean):
        w, s, suc = m.group(1), m.group(2), m.group(3)
        out.write(f'${task},${arm},${joint},locked,${mode},{s},{w},{suc}\n')
"

        line=$(grep "Final batch success rate" "$logfile" | tail -1)
        succ=$(echo "$line" | grep -oP '(?<=rate: )\d+')
        total=$(echo "$line" | grep -oP '(?<=/)\d+(?= =)')
        rate=$(echo "$line" | grep -oP '(?<== )[\d.]+(?=%)')
        echo "${task},${arm},${joint},locked,${mode},${succ:-NA},${total:-NA},${rate:-NA}" >> "$SUMMARY_CSV"
        echo "[$(date +%H:%M:%S)] -> ${succ:-?}/${total:-?} = ${rate:-?}%"
    done
done

echo "=== Phase 2 sweep complete. Results: ${RESULTS_CSV} / ${SUMMARY_CSV} ==="
