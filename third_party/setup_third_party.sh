#!/usr/bin/env bash
# Clones LIBERO and diffusion_policy at the pinned commits and applies FAVOR's joint-space changes.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C LIBERO checkout -q "$(cat third_party/LIBERO_COMMIT)"
[ -d diffusion_policy ] || git clone https://github.com/real-stanford/diffusion_policy.git
git -C diffusion_policy checkout -q "$(cat third_party/DIFFUSION_POLICY_COMMIT)"
git -C diffusion_policy apply --check ../third_party/diffusion_policy_favor.patch
git -C diffusion_policy apply ../third_party/diffusion_policy_favor.patch
echo "third_party ready"
