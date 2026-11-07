#!/usr/bin/env bash
# Train the joint-space Diffusion Policy on exported sim_xarm data.
#   sim_xarm/dp/train.sh out/xarm_pour_500.hdf5 pour500 [extra hydra overrides ...]
# Policy = upstream train_diffusion_unet_hybrid_workspace defaults, as in the LIBERO runs: CNN U-Net
# (down_dims 512/1024/2048), eps-prediction, DDPM 100 for training, horizon 16, n_obs 2, n_action 8,
# crop 76, batch 64, lr 1e-4, EMA. Like LIBERO: 100 epochs, checkpoints kept by test_mean_score
# (success of 25 no-fault rollouts, DDIM 10, every 10 epochs). Output: sim_xarm/out/train/<run_name>.
set -euo pipefail
SIM="$(cd "$(dirname "$0")/.." && pwd)"
DP="${DIFFUSION_POLICY_ROOT:-$(cd "$SIM/../.." && pwd)/diffusion_policy}"
[ $# -ge 2 ] || { echo "usage: $0 <data.hdf5> <run_name> [hydra overrides...]"; exit 1; }
DATA="$(realpath "$1")"; NAME="$2"; shift 2
[ -f "$DATA" ] || { echo "no such dataset: $DATA"; exit 1; }
export PYTHONPATH="$SIM/dp:$SIM:$DP:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset OMP_NUM_THREADS            # the data loader and torch should use all threads here
cp "$SIM/dp/xarm_pour_image_joint.yaml" "$DP/diffusion_policy/config/task/"
cd "$DP"
python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=xarm_pour_image_joint task.dataset_path="$DATA" \
  training.num_epochs=100 training.rollout_every=10 training.checkpoint_every=10 training.seed=42 \
  logging.mode=disabled exp_name="$NAME" hydra.run.dir="$SIM/out/train/$NAME" "$@"
