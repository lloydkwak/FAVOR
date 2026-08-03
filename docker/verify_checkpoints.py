"""Phase 1 DoD: all 3 checkpoints load, shape_meta present, abs_action confirmed (not assumed)."""
import sys, glob
sys.path.insert(0, "/workspace/diffusion_policy")

import torch
from omegaconf import OmegaConf, DictConfig, ListConfig

TASKS = ["lift_ph", "can_ph", "square_ph"]
FAILURES = 0

def find_key(d, key):
    # Handles both plain dict/list AND OmegaConf DictConfig/ListConfig
    if isinstance(d, (dict, DictConfig)):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(d, (list, ListConfig)):
        for v in d:
            r = find_key(v, key)
            if r is not None:
                return r
    return None

for task in TASKS:
    ckpts = glob.glob(f"/workspace/data/checkpoints/{task}/**/latest.ckpt", recursive=True)
    if not ckpts:
        print(f"[FAIL] {task}: no latest.ckpt found under data/checkpoints/{task}")
        FAILURES += 1
        continue
    ckpt_path = ckpts[0]
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg = payload.get("cfg", None)
        if cfg is None:
            print(f"[FAIL] {task}: checkpoint loaded but no 'cfg' key inside")
            FAILURES += 1
            continue
        # Normalize to plain dict for reliable, uniform inspection (source of truth,
        # not an assumption about OmegaConf's internal structure).
        cfg_dict = OmegaConf.to_container(cfg, resolve=False) if isinstance(cfg, DictConfig) else cfg

        shape_meta = find_key(cfg_dict, "shape_meta")
        abs_action = find_key(cfg_dict, "abs_action")
        n_action_steps = find_key(cfg_dict, "n_action_steps")

        print(f"[PASS] {task}: loaded {ckpt_path}")
        print(f"       shape_meta present: {shape_meta is not None}")
        if shape_meta is not None:
            print(f"       shape_meta.action.shape: {shape_meta.get('action', {}).get('shape')}")
            print(f"       shape_meta.obs keys: {list(shape_meta.get('obs', {}).keys())}")
        print(f"       abs_action (actual value found in cfg): {abs_action}")
        print(f"       n_action_steps: {n_action_steps}")
    except Exception as e:
        print(f"[FAIL] {task}: {type(e).__name__}: {e}")
        FAILURES += 1

print("=" * 72)
if FAILURES:
    print(f"{FAILURES} FAILURE(S) — Phase 1 DoD NOT met.")
    sys.exit(1)
print("ALL 3 CHECKPOINTS LOADED — check abs_action values above before Phase 2.")
