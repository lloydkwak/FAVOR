#!/usr/bin/env bash
# Sequential joint-space DP training for all 5 LIBERO tasks, num_epochs=100
# (matching LIBERO's own official convention -- ~50-100 epochs on 50
# demos/task -- rather than the 1000 epochs carried over from the
# robosuite lift/can/square runs, which was needlessly 10-20x that range
# and never got the chance to matter: three real bugs were found first
# --  upside-down training images vs. right-side-up live rendering, a
# controller gain too weak for LIBERO's floor-mounted Panda variant, and a
# gripper-action index off-by-one that fed the last joint's angle into the
# gripper channel of every cached dataset -- and are all fixed as of this
# run). rollout_every/checkpoint_every/n_test are left at the official
# config defaults (50/50/50).
set -uo pipefail

TASKS="libero_alphabet_soup libero_milk libero_bowl_ramekin libero_bowl_stove libero_drawer"

for task in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] Starting training: ${task} ==="
    docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c "
      cd /workspace/diffusion_policy && \
      python train.py \
        --config-name=train_diffusion_unet_hybrid_workspace.yaml \
        task=${task}_image_joint \
        +checkpoint=joint_topk \
        training.num_epochs=100 \
        logging.mode=offline \
        exp_name=joint_train_${task} \
        hydra.run.dir=/workspace/data/outputs/joint_train_${task}
    " > /tmp/libero_train_${task}.log 2>&1
    echo "=== [$(date +%H:%M:%S)] Finished: ${task} (exit=$?) ==="
done

echo "=== All 5 LIBERO tasks trained ==="
