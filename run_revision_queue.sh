#!/usr/bin/env bash
# Additional runs, most important first. Every step skips outputs that already exist,
# so the queue can be stopped and restarted at any point.
#   usage: ./run_revision_queue.sh [step ...]      (default: all steps below, in this order)
#   steps: method names (nofault*, wik_*, prio_rev_l001, prio_l2_*, midg_*, rg_*), ikconv, latency, qual, qual_d,
#          package, analysis
set -uo pipefail
cd "$(dirname "$0")"
DC="docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c"
TASKS="alphabet_soup milk bowl_ramekin bowl_stove"
# round 1 (done): nofault wik_w100_r000 wik_w005_r000 wik_w030_r010 wik_w020_r010 wik_w050_r010 wik_w010_r010
#                 wik_w030_r000 prio_rev_l001 rg_prioint latency qual package analysis
# round 2 (done): healthy n=100, W-IK rho=0 with Priority IK's damping, RG budget/weighting split, RG on moderate range
#          faults, failure-case video from a camera that keeps the arm in view. rg_prioint_b03 was stopped before
#          Bowl-Stove (15/20 conditions) once the budget effect was resolved; the analysis reports it as partial.
# round 3: policy-free IK convergence/residuals, Priority IK lambda_2 sweep, mid-episode (grasp-triggered) lock onset,
#          W-IK toward the lexicographic limit (w_r 0.01 / 0.001, 300 iterations)
DEFAULT="ikconv prio_l2_005 prio_l2_050 prio_l2_100 midg_b1 midg_prio midg_pos midg_pose wik_w001_r000_dm4
         wik_w001_r000_dm4_it300 wik_w0001_r000_dm4_it300 prio_l2_010 analysis"
STEPS=${*:-$DEFAULT}
filt() { stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "$1" | grep --line-buffered -v BrokenPipe; }

for s in $STEPS; do
  echo "=== [$(date '+%m-%d %H:%M')] $s ==="
  case $s in
    nofault*|wik_w*|prio_rev_l001|prio_l2_*|midg_*|rg_*)
      m=${s%%:*}; lvl=""; [[ "$s" == *:* ]] && lvl=${s#*:}
      for t in $TASKS; do
        $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
             cd /workspace/diffusion_policy && python -u /workspace/docker/run_libero_revision.py $m $t $lvl" 2>&1 \
          | filt "^RESULT|^SKIP|^DONE|Error|Traceback" || echo "!!! FAILED: $s $t"
      done ;;
    ikconv) # policy-free, CPU: solver residuals at 30 vs 300 iterations on demonstration waypoints
      $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
           cd /workspace && python -u scripts_libero/ik_convergence.py" 2>&1 | filt "^  |cells|IKCONV_DONE|Error|Traceback" ;;
    latency)
      $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
           cd /workspace/diffusion_policy && python -u /workspace/docker/bench_latency.py bowl_stove 7" 2>&1 \
        | filt "^LAT|^DONE|Error|Traceback" ;;
    qual)   # adds the C (representative proximal) and D (unrecoverable) scenarios; A and B are kept
      ./run_qual_media.sh select run compose ;;
    qual_d) # re-render only the failure case with a camera that keeps the arm in view, then recompose
      $DC "cd /workspace && python scripts_paper/select_qual_scenarios.py --results results --out results/qual/scenarios.json" \
        | grep -E "^  -> "
      $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
           cd /workspace/diffusion_policy && python -u /workspace/docker/render_qualitative.py run --only D_unrecoverable --force" 2>&1 \
        | filt "^RERUN|^DONE|render_recorder|Error|Traceback"
      ./run_qual_media.sh compose ;;
    package)
      $DC "pip install -q pillow av --break-system-packages 2>&1 | tail -1
           cd /workspace && python scripts_paper/make_preview_gif.py --video paper/video \
           && python scripts_paper/package_multimedia.py && chown -R $(id -u):$(id -g) paper" ;;
    analysis)
      python scripts_paper/make_paper_figures.py --results results --out paper | tail -2
      python scripts_paper/revision_analysis.py --results results --out paper --glmm
      python scripts_paper/export_episodes.py
      [ -f analysis_out/ik_convergence.md ] && cp analysis_out/ik_convergence.md analysis_out/ik_convergence.csv paper/analysis/ ;;
    *) echo "unknown step $s" ;;
  esac
done
echo "REVISION_QUEUE_DONE"
