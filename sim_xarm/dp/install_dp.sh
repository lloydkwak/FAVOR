#!/usr/bin/env bash
# Installs the training stack into the CURRENT python env (activate sim_xarm first):
# upstream diffusion_policy at a pinned commit (next to the FAVOR checkout, or $DIFFUSION_POLICY_ROOT),
# the pinned python deps, and the sim_xarm task config. Safe to re-run.
set -euo pipefail
SIM="$(cd "$(dirname "$0")/.." && pwd)"
DP="${DIFFUSION_POLICY_ROOT:-$(cd "$SIM/../.." && pwd)/diffusion_policy}"
DP_COMMIT=5ba07ac6661db573af695b419a7947ecb704690f     # upstream real-stanford/diffusion_policy (main, 2024)
if [ ! -d "$DP/.git" ]; then
  git clone https://github.com/real-stanford/diffusion_policy.git "$DP"
fi
git -C "$DP" fetch -q origin "$DP_COMMIT" 2>/dev/null || true
git -C "$DP" checkout -q "$DP_COMMIT"
python -c "import torchvision" 2>/dev/null || pip install torchvision
pip install -r "$SIM/dp/requirements_dp.txt"
pip install --no-deps robomimic==0.2.0
cp "$SIM/dp/xarm_pour_image_joint.yaml" "$DP/diffusion_policy/config/task/"
cd "$SIM"
PYTHONPATH="$SIM/dp:$SIM:$DP" python -W ignore - <<'PY'
import torch, torchvision
import diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace  # noqa: F401
import xarm_pour_runner, xarm_pour_dataset  # noqa: F401
print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, CUDA available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)"
         if torch.cuda.is_available() else ""))
print("diffusion_policy + sim_xarm imports OK")
PY
echo "diffusion_policy at $DP (commit $DP_COMMIT). Train with: sim_xarm/dp/train.sh <data.hdf5> <run_name>"
