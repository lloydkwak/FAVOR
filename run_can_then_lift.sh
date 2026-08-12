#!/bin/bash
# Waits for the Square docker container to disappear from `docker ps`
# (finished or crashed), checking container NAME rather than a host PID --
# more reliable than kill -0 on a PID that lives inside a container's
# namespace (a prior attempt using kill -0 9465 failed silently: Can
# started immediately even though Square was still running, causing GPU
# contention that had to be manually cleaned up).
set -x
cd ~/favor_project

SQUARE_CONTAINER_NAME_PATTERN="docker-favor-run"

echo "Waiting for Square training container to exit..."
while docker ps --format '{{.Names}}' | grep -q "$SQUARE_CONTAINER_NAME_PATTERN"; do
    sleep 30
done
echo "No favor-run container detected. Square training has ended. Proceeding to Can."

CAN_DIR=/workspace/data/outputs/joint_train_can_run
LIFT_DIR=/workspace/data/outputs/joint_train_lift_run

run_task() {
    local task=$1
    local dataset_path=$2
    local run_dir=$3
    echo "=================================================="
    echo "STARTING: $task  (run_dir=$run_dir)"
    echo "=================================================="
    docker compose -f docker/docker-compose.yml run --rm -e PYTHONPATH=/workspace/docker favor bash -c "
    git config --global --add safe.directory /workspace &&
    python diffusion_policy/train.py \
        --config-name=train_diffusion_unet_hybrid_workspace.yaml \
        task=${task}_image_joint \
        task.dataset.dataset_path=${dataset_path} \
        task.env_runner.dataset_path=${dataset_path} \
        +checkpoint=joint_topk \
        training.checkpoint_every=5 \
        training.rollout_every=5 \
        training.resume=true \
        logging.mode=offline \
        hydra.run.dir=${run_dir} \
        exp_name=joint_train_${task}
    "
    echo "=================================================="
    echo "EXITED: $task"
    echo "=================================================="
}

run_task can  /workspace/diffusion_policy/data/robomimic/datasets/can/ph/image_abs.hdf5  "$CAN_DIR"
run_task lift /workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5 "$LIFT_DIR"

echo "CAN AND LIFT COMPLETE (or exited)."
