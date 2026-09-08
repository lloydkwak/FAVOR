#!/usr/bin/env bash
# Sequential joint-space DP training for all 5 LIBERO tasks, matching the
# robosuite lift/can/square runs: num_epochs=1000, official rollout_every=50
# / checkpoint_every=50 / n_test=50 (left at config defaults, not overridden
# here). Runs one task at a time so each gets the full GPU.
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
        training.num_epochs=1000 \
        logging.mode=offline \
        exp_name=joint_train_${task} \
        hydra.run.dir=/workspace/data/outputs/joint_train_${task}
    " > /tmp/libero_train_${task}.log 2>&1
    echo "=== [$(date +%H:%M:%S)] Finished: ${task} (exit=$?) ==="
done

echo "=== All 5 LIBERO tasks trained ==="
