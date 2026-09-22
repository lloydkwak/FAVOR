#!/usr/bin/env bash
set -uo pipefail
cd ~/favor_project
DR=/workspace/data/outputs/joint_train_libero_drawer/checkpoints

run_eval () {  # task ckpt kp
  local log=/tmp/eval_$1_$3_$(basename "$2" .ckpt)_$(date +%H%M).log
  docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c "
    pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
    cd /workspace/diffusion_policy && python -u /workspace/scripts_libero/eval_nofault.py $1 '$2' $3 20
  " > "$log" 2>&1
  grep -o "RESULT.*" "$log" || echo "NO RESULT: $1 kp=$3 $2 (log: $log)"
}

echo "=== [$(date +%H:%M)] 1) drawer epoch-50 ckpt, kp=1000 ==="
run_eval drawer "$DR/latest.ckpt" 1000

echo "=== [$(date +%H:%M)] 2) drawer resume training from epoch 50 ==="
docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c "
  cd /workspace/diffusion_policy && python train.py \
    --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=libero_drawer_image_joint \
    +checkpoint=joint_topk \
    training.num_epochs=101 \
    training.resume=true \
    logging.mode=offline \
    exp_name=joint_train_libero_drawer \
    hydra.run.dir=/workspace/data/outputs/joint_train_libero_drawer
" > /tmp/libero_resume_drawer.log 2>&1
echo "resume exit=$?"
ls -la ~/favor_project/data/outputs/joint_train_libero_drawer/checkpoints/

echo "=== [$(date +%H:%M)] 3) drawer final ckpt, kp=600 / 1000 ==="
run_eval drawer "$DR/latest.ckpt" 600
run_eval drawer "$DR/latest.ckpt" 1000
echo "=== [$(date +%H:%M)] done ==="
