#!/usr/bin/env bash
# Every main-result sweep (results/...). One container per (runner, task); finished conditions are skipped.
# Range level index: 0 = moderate, 1 = severe, 2 = mild.
set -uo pipefail
cd "$(dirname "$0")"
DC="docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c"
TASKS="alphabet_soup milk bowl_ramekin bowl_stove"
run () {
  echo "=== [$(date '+%m-%d %H:%M')] $* ==="
  $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
       cd /workspace/diffusion_policy && python -u /workspace/docker/$*" 2>&1 \
    | stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "^RESULT|^SKIP|^DONE|Error" | grep --line-buffered -v BrokenPipe \
    || echo "!!! FAILED: $*"
}
for t in $TASKS; do
  run run_libero_fault_sweep_locked.py $t b1
  run run_libero_fault_sweep_eci.py $t
  run run_libero_fault_sweep_phase2_random_n.py $t
  run run_libero_fault_sweep_phase2_select.py $t
  run run_libero_fault_sweep_locked.py $t ik
  run run_libero_fault_sweep_locked.py $t ik_pose
  run run_libero_fault_sweep_prio.py $t locked
  for lv in 0 1 2; do
    for m in b1 eci random_n ik ik_pose; do run run_libero_fault_sweep_range.py $t $lv $m; done
    run run_libero_fault_sweep_prio.py $t range $lv
  done
  run run_libero_fault_sweep_range.py $t 0 select
done
echo "MAIN_SWEEPS_DONE"
