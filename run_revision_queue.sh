#!/usr/bin/env bash
# Reviewer-requested runs, most important first. Every step skips outputs that already exist,
# so the queue can be stopped and restarted at any point.
#   usage: ./run_revision_queue.sh [step ...]      (default: all steps below, in this order)
#   steps: nofault, wik_* / prio_rev_l001 / rg_prioint (method names), latency, qual, package, analysis
set -uo pipefail
cd "$(dirname "$0")"
DC="docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c"
TASKS="alphabet_soup milk bowl_ramekin bowl_stove"
DEFAULT="nofault wik_w100_r000 wik_w005_r000 wik_w030_r010 wik_w020_r010 wik_w050_r010 wik_w010_r010 wik_w030_r000 \
prio_rev_l001 rg_prioint latency qual package analysis"
STEPS=${*:-$DEFAULT}
filt() { stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "$1" | grep --line-buffered -v BrokenPipe; }

for s in $STEPS; do
  echo "=== [$(date '+%m-%d %H:%M')] $s ==="
  case $s in
    nofault|wik_w*|prio_rev_l001|rg_prioint)
      for t in $TASKS; do
        $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
             cd /workspace/diffusion_policy && python -u /workspace/docker/run_libero_revision.py $s $t" 2>&1 \
          | filt "^RESULT|^SKIP|^DONE|Error|Traceback" || echo "!!! FAILED: $s $t"
      done ;;
    latency)
      $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
           cd /workspace/diffusion_policy && python -u /workspace/docker/bench_latency.py bowl_stove 7" 2>&1 \
        | filt "^LAT|^DONE|Error|Traceback" ;;
    qual)   # adds the C (representative proximal) and D (unrecoverable) scenarios; A and B are kept
      ./run_qual_media.sh select run compose ;;
    package)
      $DC "pip install -q pillow av --break-system-packages 2>&1 | tail -1
           cd /workspace && python scripts_paper/make_preview_gif.py --video paper/video \
           && python scripts_paper/package_multimedia.py && chown -R $(id -u):$(id -g) paper" ;;
    analysis)
      python scripts_paper/make_paper_figures.py --results results --out paper | tail -2
      python scripts_paper/revision_analysis.py --results results --out paper --glmm
      python scripts_paper/export_episodes.py ;;
    *) echo "unknown step $s" ;;
  esac
done
echo "REVISION_QUEUE_DONE"
