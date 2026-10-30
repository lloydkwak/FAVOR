#!/usr/bin/env bash
# Fetches the MuJoCo Menagerie at the pinned commit into sim_xarm/third_party/ (not tracked).
# The xArm7 URDF used for FK/IK is generated and tracked (assets/xarm7_favor.urdf), so xarm_ros2
# is only needed to regenerate it (scripts/make_urdf.py --xarm-ros2 <path>).
set -euo pipefail
cd "$(dirname "$0")"
MENAGERIE_COMMIT=f3b6013629c7b94f8ee93fd977b161fa8350f9bc
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
fi
git -C third_party/mujoco_menagerie fetch -q origin "$MENAGERIE_COMMIT" || true
git -C third_party/mujoco_menagerie checkout -q "$MENAGERIE_COMMIT"
git -C third_party/mujoco_menagerie sparse-checkout set ufactory_xarm7 robotiq_2f85 2>/dev/null || true
test -f third_party/mujoco_menagerie/ufactory_xarm7/xarm7.xml
echo "menagerie ready at $MENAGERIE_COMMIT"
