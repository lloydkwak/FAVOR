"""diffusion_policy dataset for the sim_xarm pouring data (robomimic-layout hdf5 written by
sim_xarm/scripts/m5_export.py).

Why not RobomimicReplayImageDataset: with abs_action=False it gives the action an IDENTITY
normalizer (robomimic delta actions are pre-normalised), and with abs_action=True it treats the
action as pos + axis-angle and converts it to rot6d. Neither fits absolute joint targets, so this
class mirrors its replay-buffer / sampler / normalizer logic (diffusion_policy, MIT license) with:
  - action = the 7 commanded joint targets, range-normalised to [-1, 1];
  - low-dim obs: *pos range-normalised, *quat identity (as upstream);
  - rgb obs: image range normaliser; images Jpeg2k-compressed in the zarr cache (as upstream).
It does not import pytorch3d. Put sim_xarm/dp on PYTHONPATH and use
  _target_: xarm_pour_dataset.XArmPourImageDataset
"""
import copy
import os
import shutil
from typing import Dict

import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits

from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from diffusion_policy.common.normalize_util import (array_to_stats, get_identity_normalizer_from_stat,
                                                   get_image_range_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

register_codecs()


def _split_keys(shape_meta):
    rgb, low = [], []
    for k, a in shape_meta["obs"].items():
        (rgb if a.get("type", "low_dim") == "rgb" else low).append(k)
    return rgb, low


def convert_to_replay(store, shape_meta, dataset_path):
    rgb_keys, low_keys = _split_keys(shape_meta)
    root = zarr.group(store)
    data = root.require_group("data", overwrite=True)
    meta = root.require_group("meta", overwrite=True)
    with h5py.File(dataset_path, "r") as f:
        demos = f["data"]
        n = len(demos)
        lens = [demos[f"demo_{i}"]["actions"].shape[0] for i in range(n)]
        ends = np.cumsum(lens)
        meta.array("episode_ends", ends, dtype=np.int64, compressor=None, overwrite=True)
        act = np.concatenate([demos[f"demo_{i}"]["actions"][:] for i in range(n)]).astype(np.float32)
        assert act.shape[1:] == tuple(shape_meta["action"]["shape"]), (act.shape, shape_meta["action"]["shape"])
        data.array("action", act, chunks=act.shape, compressor=None, dtype=act.dtype)
        for k in low_keys:
            x = np.concatenate([demos[f"demo_{i}"]["obs"][k][:] for i in range(n)]).astype(np.float32)
            assert x.shape[1:] == tuple(shape_meta["obs"][k]["shape"]), (k, x.shape)
            data.array(k, x, chunks=x.shape, compressor=None, dtype=x.dtype)
        for k in rgb_keys:
            c, h, w = shape_meta["obs"][k]["shape"]
            arr = data.require_dataset(k, shape=(int(ends[-1]), h, w, c), chunks=(1, h, w, c),
                                       compressor=Jpeg2k(level=50), dtype=np.uint8)
            s = 0
            for i in range(n):
                x = demos[f"demo_{i}"]["obs"][k][:]
                assert x.shape[1:] == (h, w, c), (k, x.shape, "images must be HWC uint8")
                arr[s:s + len(x)] = x
                s += len(x)
    return ReplayBuffer(root)


class XArmPourImageDataset(BaseImageDataset):
    def __init__(self, shape_meta: dict, dataset_path: str, horizon=1, pad_before=0, pad_after=0,
                 n_obs_steps=None, use_cache=False, seed=42, val_ratio=0.0, **unused):
        if use_cache:
            cache = dataset_path + ".zarr.zip"
            with FileLock(cache + ".lock"):
                if not os.path.exists(cache):
                    try:
                        rb = convert_to_replay(zarr.MemoryStore(), shape_meta, dataset_path)
                        with zarr.ZipStore(cache) as zs:
                            rb.save_to_store(store=zs)
                    except Exception:
                        if os.path.exists(cache):
                            os.remove(cache) if os.path.isfile(cache) else shutil.rmtree(cache)
                        raise
                else:
                    with zarr.ZipStore(cache, mode="r") as zs:
                        rb = ReplayBuffer.copy_from_store(src_store=zs, store=zarr.MemoryStore())
        else:
            rb = convert_to_replay(zarr.MemoryStore(), shape_meta, dataset_path)
        self.rgb_keys, self.lowdim_keys = _split_keys(shape_meta)
        key_first_k = {k: n_obs_steps for k in self.rgb_keys + self.lowdim_keys} if n_obs_steps is not None else {}
        val_mask = get_val_mask(n_episodes=rb.n_episodes, val_ratio=val_ratio, seed=seed)
        self.train_mask = ~val_mask
        self.sampler = SequenceSampler(replay_buffer=rb, sequence_length=horizon, pad_before=pad_before,
                                       pad_after=pad_after, episode_mask=self.train_mask, key_first_k=key_first_k)
        self.replay_buffer = rb
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.horizon, self.pad_before, self.pad_after = horizon, pad_before, pad_after
        self.key_first_k = key_first_k

    def get_validation_dataset(self):
        v = copy.copy(self)
        v.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon,
                                    pad_before=self.pad_before, pad_after=self.pad_after,
                                    episode_mask=~self.train_mask, key_first_k=self.key_first_k)
        v.train_mask = ~self.train_mask
        return v

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(array_to_stats(self.replay_buffer["action"]))
        for k in self.lowdim_keys:
            st = array_to_stats(self.replay_buffer[k])
            if k.endswith("quat"):
                normalizer[k] = get_identity_normalizer_from_stat(st)
            else:
                normalizer[k] = get_range_normalizer_from_stat(st)
        for k in self.rgb_keys:
            normalizer[k] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        T = slice(self.n_obs_steps)
        obs = {}
        for k in self.rgb_keys:
            obs[k] = np.moveaxis(data[k][T], -1, 1).astype(np.float32) / 255.0     # T,H,W,C -> T,C,H,W
            del data[k]
        for k in self.lowdim_keys:
            obs[k] = data[k][T].astype(np.float32)
            del data[k]
        return dict_apply({"obs": obs, "action": data["action"].astype(np.float32)}, torch.from_numpy)
