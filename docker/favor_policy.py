"""
FavorHybridImagePolicy — wraps an already-loaded DiffusionUnetHybridImagePolicy
without modifying the official file. Overrides only conditional_sample() to
insert the E-C-I hook; predict_action() is a byte-for-byte copy of the
official method (needed because it calls self.conditional_sample internally,
so we can't just monkeypatch one method on the instance without also owning
predict_action's call site).

Phase 4-1: fault_spec=None (default) => E-C-I is skipped entirely, trajectory
is exactly scheduler.step(...).prev_sample every iteration — bit-identical to
B1. This is not a placeholder shortcut; it also means FAVOR does zero extra
work in the fault-free case (relevant to Phase 4-2's "무고장 성공률 유지" DoD).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
import torch
from diffusion_policy.common.pytorch_util import dict_apply


class FavorHybridImagePolicy:
    def __init__(self, base_policy, fault_spec=None, projector=None, env_ref=None, blend_floor=0.0):
        self.base = base_policy
        self.fault_spec = fault_spec      # static part: joint_idx / fault_type / severity (same for all envs in one runner)
        self.projector = projector        # Phase 4-2+: callable(raw_clean, fault_spec, q_prev_seed) -> raw_corrected
        self._q_prev_seed = None          # warm-start state across waypoints/steps within an episode
        self.env_ref = env_ref            # AsyncVectorEnv reference -- used to pull PER-ENV q_lock live via RPC,
                                           # since the actual lock value is only known after each env's own
                                           # random reset() and differs across envs (see FaultInjector.get_fault_info)
        self.blend_floor = blend_floor    # min correction weight even in early (high-noise) denoising steps.
                                           # Motivation (empirically measured, not from literature -- no precedent
                                           # for this bimodal/floored schedule): with pure sqrt(alpha_bar_t), the
                                           # first ~10 steps (where the trajectory's overall shape gets decided)
                                           # get blend_frac < 0.14, i.e. almost no correction, so the policy commits
                                           # to a "pretend nothing is locked" trajectory shape before FAVOR gets a
                                           # meaningful say. Default 0.0 preserves the original literature-aligned
                                           # schedule exactly (backward compatible).

    # -- passthrough attributes used by RobomimicImageRunner's run() loop --
    @property
    def device(self):
        return self.base.device

    @property
    def dtype(self):
        return self.base.dtype

    def to(self, device):
        self.base.to(device)
        return self

    def eval(self):
        self.base.eval()
        return self

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()
        self._q_prev_seed = None  # cleared at episode start; projector will seed from zeros on first call

    # -- E-C-I hook --
    def conditional_sample(self, condition_data, condition_mask,
                            local_cond=None, global_cond=None,
                            generator=None, **kwargs):
        base = self.base
        model = base.model
        scheduler = base.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype,
            device=condition_data.device, generator=generator)

        scheduler.set_timesteps(base.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t,
                                  local_cond=local_cond, global_cond=global_cond)
            step_out = scheduler.step(model_output, t, trajectory,
                                       generator=generator, **kwargs)
            prev_sample = step_out.prev_sample

            if self.fault_spec is None:
                # Phase 4-1 DoD: identical to vanilla B1, no E-C-I overhead.
                trajectory = prev_sample
                continue

            # === E: Tweedie estimate, already computed by the scheduler ===
            clean_est = step_out.pred_original_sample

            # === C: unnormalize -> project -> renormalize ===
            raw_clean = base.normalizer["action"].unnormalize(clean_est)
            if self.projector is None:
                raw_corrected = raw_clean            # Phase 4-1: identity
            else:
                import torch as _torch
                B = raw_clean.shape[0]
                if self._q_prev_seed is None:
                    self._q_prev_seed = _torch.zeros(B, 7, dtype=raw_clean.dtype, device=raw_clean.device)

                per_env_fault_spec = dict(self.fault_spec)
                if self.env_ref is not None:
                    # Pull the REAL, per-environment lock value live -- each env's
                    # random reset() produced a different onset value (verified
                    # empirically: -2.644/-2.646/-2.624 across 3 resets of the
                    # same joint). A single hardcoded q_lock would silently solve
                    # the wrong problem for every env except by coincidence.
                    infos = self.env_ref.call('get_fault_info')
                    q_locks = [info['q_lock'] if info['q_lock'] is not None else 0.0 for info in infos]
                    per_env_fault_spec['q_lock'] = _torch.tensor(q_locks, dtype=raw_clean.dtype, device=raw_clean.device)

                raw_corrected = self.projector(raw_clean, per_env_fault_spec, self._q_prev_seed)
                self._q_prev_seed = self.projector.last_q_seed
            norm_corrected = base.normalizer["action"].normalize(raw_corrected)

            # === I: schedule-gated blend ===
            # Direction verified against literature (CPS/DiRecT/Feynman-Kac,
            # see FAVOR_이론적_타당성_분석.md §3): correction weight increases
            # toward the END of denoising (alpha_prod_t -> 1), not the start.
            alpha_prod_t = scheduler.alphas_cumprod[t]
            blend_frac = torch.clamp(torch.sqrt(alpha_prod_t), min=self.blend_floor)
            trajectory = (1 - blend_frac) * prev_sample + blend_frac * norm_corrected

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # -- byte-for-byte copy of DiffusionUnetHybridImagePolicy.predict_action,
    #    except it calls self.conditional_sample (ours) instead of base's --
    def predict_action(self, obs_dict):
        base = self.base
        assert "past_action" not in obs_dict
        nobs = base.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = base.horizon
        Da = base.action_dim
        Do = base.obs_feature_dim
        To = base.n_obs_steps
        device = base.device
        dtype = base.dtype

        local_cond = None
        global_cond = None
        if base.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = base.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = base.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=local_cond, global_cond=global_cond,
            **base.kwargs)

        naction_pred = nsample[..., :Da]
        action_pred = base.normalizer["action"].unnormalize(naction_pred)
        start = To - 1
        end = start + base.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}
