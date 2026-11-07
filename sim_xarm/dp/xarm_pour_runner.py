"""Env runners for the sim_xarm pouring task inside diffusion_policy training.

PourEnvRunner: no-fault rollouts every `training.rollout_every` epochs on fixed seeds (one per grid
cell by default), logged as test_mean_score (success rate: f >= 0.8 and no safety violation), so
the upstream checkpoint.topk (monitor_key test_mean_score, mode max, k 5) selects checkpoints by
task success, as in the LIBERO runs (favor_fault_runner, n_test 50). Uses the inference sampler
of the evaluation (DDIM 10 by default, the real-robot plan).
NullRunner: no rollouts (checkpoint on train_loss).
"""
import os
import sys

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)


class NullRunner(BaseImageRunner):
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(output_dir)

    def run(self, policy):
        return {}


class PourEnvRunner(BaseImageRunner):
    def __init__(self, output_dir=None, shape_meta=None, n_obs_steps=2, n_action_steps=8, n_test=25,
                 test_start_seed=100000, n_workers=8, sampler="ddim", inference_steps=10, max_steps=300,
                 config_overrides=(), early_stop=True, **kwargs):
        super().__init__(output_dir)
        from favor_xarm.config import load_config
        self.cfg = load_config(overrides=list(config_overrides))
        self.cfg["task"]["max_steps"] = int(max_steps)
        self.obs_keys = list(shape_meta["obs"].keys())
        self.n_obs, self.n_act = int(n_obs_steps), int(n_action_steps)
        n_cells = self.cfg["task"]["grid_n"] ** 2
        self.episodes = [dict(seed=test_start_seed + i, grid_index=i % n_cells, fault=None, method="B1")
                         for i in range(int(n_test))]
        self.n_workers = min(int(n_workers), int(n_test))
        self.sampler, self.steps, self.early_stop = sampler, inference_steps, early_stop
        self.pool = None

    def run(self, policy):
        import copy
        import numpy as np
        from favor_xarm.policy_rollout import RolloutPool, run_episodes, set_sampler
        if self.pool is None:
            self.pool = RolloutPool(self.cfg, self.n_workers, self.n_obs, early_stop=self.early_stop)
        # Sample with a copy so the training model's scheduler is untouched.
        pol = set_sampler(copy.copy(policy), self.sampler, self.steps)
        res = run_episodes(pol, self.pool, self.episodes, self.obs_keys, self.n_act, kin=None, log=None)
        s = np.array([r["success"] for r in res], float)
        f = np.array([r["f"] for r in res], float)
        return {"test_mean_score": float(s.mean()), "test/mean_f": float(f.mean()),
                "test/mean_spill_g": float(np.mean([r["spill_g"] for r in res])),
                "test/safety_violation_rate": float(np.mean([r["safety_violation"] for r in res]))}

    def __del__(self):
        if getattr(self, "pool", None) is not None:
            self.pool.close()
