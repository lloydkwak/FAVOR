#!/usr/bin/env bash
set -uo pipefail
cd ~/favor_project
while tmux has-session -t drawer_chain 2>/dev/null; do sleep 60; done
echo "=== [$(date +%H:%M)] drawer_chain finished, starting bowl_stove sweeps ==="

for script in run_libero_fault_sweep_phase1 run_libero_fault_sweep_phase2_select \
              run_libero_fault_sweep_phase2_random_n run_libero_fault_sweep_eci; do
  echo "=== [$(date +%H:%M)] ${script} bowl_stove ==="
  docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c "
    pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
    cd /workspace/diffusion_policy && python -u /workspace/docker/${script}.py bowl_stove
  " > /tmp/sweep_${script}_bowl_stove.log 2>&1
  echo "exit=$?"
  grep -o "\[favor_fault_runner\][^\[]*" /tmp/sweep_${script}_bowl_stove.log | sort | uniq -c
  grep -oE "bowl_stove/robot0_joint[0-9]/locked/None/[a-z_0-9]+: [0-9.]+" /tmp/sweep_${script}_bowl_stove.log
done
echo "=== [$(date +%H:%M)] all bowl_stove sweeps done ==="
