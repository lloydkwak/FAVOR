"""M5 check: the exported hdf5 loads through the diffusion_policy data pipeline with the task config
sim_xarm/dp/xarm_pour_image_joint.yaml (horizon 16, n_obs 2, n_action 8 as in the LIBERO runs).
Needs the diffusion_policy repo on PYTHONPATH (DIFFUSION_POLICY_ROOT or ../diffusion_policy).
Run: python sim_xarm/tests/test_dp_dataset.py <exported.hdf5>"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dp"))
dp_root = os.environ.get("DIFFUSION_POLICY_ROOT", os.path.join(HERE, "..", "..", "diffusion_policy"))
sys.path.insert(0, dp_root)
import numpy as np
import torch
import yaml

from xarm_pour_dataset import XArmPourImageDataset


def main():
    path = sys.argv[1]
    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "dp", "xarm_pour_image_joint.yaml")))
    shape_meta = cfg["shape_meta"]
    horizon, n_obs, n_act = 16, 2, 8
    for p in (path + ".zarr.zip", path + ".zarr.zip.lock"):
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()
    ds = XArmPourImageDataset(shape_meta=shape_meta, dataset_path=path, horizon=horizon, pad_before=n_obs - 1,
                              pad_after=n_act - 1, n_obs_steps=n_obs, use_cache=True, seed=42, val_ratio=0.02)
    t_build = time.time() - t0
    ds2 = XArmPourImageDataset(shape_meta=shape_meta, dataset_path=path, horizon=horizon, pad_before=n_obs - 1,
                               pad_after=n_act - 1, n_obs_steps=n_obs, use_cache=True, seed=42, val_ratio=0.02)
    norm = ds.get_normalizer()
    val = ds.get_validation_dataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    b = next(iter(dl))
    shapes = {k: tuple(v.shape) for k, v in b["obs"].items()}
    nobs = norm.normalize(b["obs"])            # as the policy does: obs dict, then the action field
    a = norm["action"].normalize(b["action"])
    assert all(torch.isfinite(v).all() for v in nobs.values())
    ok = (shapes["agentview_image"] == (64, n_obs, 3, 84, 84) and shapes["robot0_eye_in_hand_image"] == (64, n_obs, 3, 84, 84)
          and shapes["robot0_eef_pos"] == (64, n_obs, 3) and shapes["robot0_eef_quat"] == (64, n_obs, 4)
          and tuple(b["action"].shape) == (64, horizon, 7) and float(a.abs().max()) <= 1.0 + 1e-5
          and float(b["obs"]["agentview_image"].max()) <= 1.0 and len(ds2) == len(ds))
    n_batches = 0
    for bb in dl:
        n_batches += 1
    act = ds.replay_buffer["action"][:]
    print(f"episodes {ds.replay_buffer.n_episodes}, steps {ds.replay_buffer.n_steps}, train samples {len(ds)}, "
          f"val samples {len(val)}, cache build {t_build:.1f} s")
    print("batch obs shapes", shapes, "action", tuple(b["action"].shape))
    print(f"normalised action range [{float(a.min()):.3f}, {float(a.max()):.3f}]; raw action range per joint "
          f"{np.round(act.min(0), 2).tolist()} .. {np.round(act.max(0), 2).tolist()}")
    print(f"full epoch iterated: {n_batches} batches")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
