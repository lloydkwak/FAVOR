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
    def __init__(self, base_policy, fault_spec=None, projector=None, env_ref=None, blend_floor=0.0,
                 actuation_mode='osc', joint_q_lo=None, joint_q_hi=None,
                 joint_output_projector_K=64, joint_output_projector_iters=5,
                 joint_output_projector_lambda=0.3):
        self.base = base_policy
        self.fault_spec = fault_spec      # static part: joint_idx / fault_type / severity (same for all envs in one runner)
        self.projector = projector        # Phase 4-2+: callable(raw_clean, fault_spec, q_prev_seed) -> raw_corrected
                                           # (mid-DENOISING C-step correction -- unchanged, still operates on noisy
                                           # intermediate estimates, EE-pose space, exactly as before)
        self._q_prev_seed = None          # warm-start state across waypoints/steps within an episode (mid-denoising C-step)
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

        # --- actuation_mode: kept STRICTLY separate from everything above. ---
        # 'osc'   (default) -> predict_action returns the EE-pose action exactly as
        #          before (bit-identical code path, verified by test_favor_identity.py).
        # 'joint' -> after the (unchanged) EE-pose trajectory is produced, ONE extra
        #          terminal IK pass converts it to joint targets for JointActuationWrapper.
        #          This is a SEPARATE ProjectWaypoints instance/seed chain from
        #          self.projector (the mid-denoising C-step) -- deliberately decoupled,
        #          per explicit instruction, so neither can silently affect the other.
        self.actuation_mode = actuation_mode
        if actuation_mode == 'joint':
            assert joint_q_lo is not None and joint_q_hi is not None, \
                "actuation_mode='joint' requires joint_q_lo/joint_q_hi (needed for the terminal IK pass even when fault_spec is None)"
            from ik_projector import ProjectWaypoints
            self._joint_output_projector = ProjectWaypoints(
                K=joint_output_projector_K, iters=joint_output_projector_iters,
                lambda_reg=joint_output_projector_lambda)
            self._joint_q_lo = joint_q_lo
            self._joint_q_hi = joint_q_hi
            self._q_prev_seed_joint_output = None   # separate warm-start chain, terminal IK pass only

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
        if self.actuation_mode == 'joint':
            self._q_prev_seed_joint_output = None  # separate chain, cleared independently

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
                    if self.env_ref is not None:
                        # Seed with the ROBOT'S ACTUAL current joint config,
                        # not an arbitrary zero vector -- critical now that
                        # C uses joint-space regularization toward q_ref.
                        qpos_list = self.env_ref.call('get_current_qpos')
                        self._q_prev_seed = _torch.tensor(
                            [list(q) for q in qpos_list], dtype=raw_clean.dtype, device=raw_clean.device)
                    else:
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

        if self.actuation_mode == 'osc':
            # UNCHANGED path -- identical to before this patch, byte-for-byte.
            return {"action": action, "action_pred": action_pred}

        # --- actuation_mode == 'joint': terminal IK pass, EE-pose -> joint targets ---
        # This runs regardless of self.fault_spec (even when None / no fault), because
        # JointActuationWrapper needs joint targets unconditionally -- there is no OSC
        # to fall back on. When self.fault_spec is None we pass a fault_type=None spec
        # (locked_mask stays all-False inside the projector), which reduces this to a
        # plain unconstrained IK solve -- reusing the exact same, already-verified
        # projector code, not a new IK path.
        import torch as _torch
        B = action.shape[0]
        # ALWAYS re-anchor the IK seed to the robot's ACTUAL current joint
        # config (not just on the first call). Root cause found empirically:
        # chaining IK's own previous solution across predict_action calls
        # silently assumes the robot already reached that solution -- but the
        # low-level joint controller only gets ~8 substeps to catch up, so
        # the IK chain drifts further and further from where the robot
        # actually is (confirmed: chain-to-chain jump stayed small, ~0.14-0.23
        # rad, while actual-robot-to-target gap grew to ~2 rad and never
        # closed even with the speed budget effectively unlimited).
        if self.env_ref is not None:
            qpos_list = self.env_ref.call('get_current_qpos')
            self._q_prev_seed_joint_output = _torch.tensor(
                [list(q) for q in qpos_list], dtype=action.dtype, device=action.device)
        elif self._q_prev_seed_joint_output is None:
            self._q_prev_seed_joint_output = _torch.zeros(B, 7, dtype=action.dtype, device=action.device)

        if self.fault_spec is not None:
            terminal_fault_spec = dict(self.fault_spec)
            if self.env_ref is not None:
                infos = self.env_ref.call('get_fault_info')
                q_locks = [info['q_lock'] if info['q_lock'] is not None else 0.0 for info in infos]
                terminal_fault_spec['q_lock'] = _torch.tensor(q_locks, dtype=action.dtype, device=action.device)
        else:
            terminal_fault_spec = {
                'joint_idx': 0, 'q_lock': 0.0, 'fault_type': None,
                'q_lo': self._joint_q_lo, 'q_hi': self._joint_q_hi,
            }
        # ensure q_lo/q_hi present even if fault_spec was supplied without them
        terminal_fault_spec.setdefault('q_lo', self._joint_q_lo)
        terminal_fault_spec.setdefault('q_hi', self._joint_q_hi)

        # action[..., 0:9] = pos(3)+rot6d(6); action[...,9] = gripper -- kept aside, reattached after IK.
        from ik_projector import project_waypoints_to_joint_targets
        gripper_col = action[..., 9:10]
        q_targets, final_q_seed = project_waypoints_to_joint_targets(
            action[..., 0:9], terminal_fault_spec, self._q_prev_seed_joint_output,
            K=self._joint_output_projector.K, iters=self._joint_output_projector.iters,
            lambda_reg=self._joint_output_projector.lambda_reg)
        self._q_prev_seed_joint_output = final_q_seed

        joint_action = _torch.cat([q_targets, gripper_col], dim=-1)  # (B, n_action_steps, 8)
        if getattr(self, "_debug_gripper", False):
            print("[favor_policy debug] raw gripper_col (pre any wrapper):", gripper_col[0].squeeze(-1).detach().cpu().numpy())
        return {"action": joint_action, "action_pred": action_pred}
