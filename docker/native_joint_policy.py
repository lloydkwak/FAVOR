"""
Native joint-space policy wrapper for actuation_mode='joint' evaluation,
used by both B1 (no correction) and FAVOR (E-C-I / post-hoc projection)
conditions. Replaces the EE-pose-era FavorHybridImagePolicy for checkpoints
retrained natively in joint space (no IK anywhere in this module).

B1: fault_spec=None -> conditional_sample is the ORIGINAL, untouched
    denoising loop (policy.conditional_sample) -- zero risk of any subtle
    difference from a "pure" baseline, since we don't even call our own
    eci_conditional_sample in this case.
FAVOR (E-C-I): fault_spec set, mode='eci' -> eci_conditional_sample runs
    project+renoise at every denoising step (see joint_eci_projector.py).
FAVOR (post-hoc): fault_spec set, mode='posthoc' -> ORIGINAL unconstrained
    denoising runs to completion, then project_fault is applied ONCE to the
    final x0 -- no renoise (there's no more denoising left to do, so PPR's
    manifold-recovery step doesn't apply here by construction; this mode
    exists purely as the "naive projection" baseline for comparison against
    eci mode, mirroring this session's earlier finding that these two were
    indistinguishable under the OLD EE-pose+IK pipeline -- to be re-tested
    now that IK-induced distortion is no longer a confound).
"""
import torch
import sys
sys.path.insert(0, '/workspace/diffusion_policy')
from diffusion_policy.common.pytorch_util import dict_apply
from joint_eci_projector import eci_conditional_sample, project_fault


class NativeJointPolicy:
    def __init__(self, base_policy, fault_spec=None, mode='eci'):
        """
        base_policy: the loaded DiffusionUnetHybridImagePolicy (native
                     joint-space checkpoint), already .to(device).eval()'d.
        fault_spec: None for B1, or a dict (see joint_eci_projector.project_fault)
                    for FAVOR.
        mode: 'eci' or 'posthoc' -- ignored if fault_spec is None.
        """
        assert mode in ('eci', 'posthoc')
        self.base = base_policy
        self.fault_spec = fault_spec
        self.mode = mode
        self.device = base_policy.device
        self.dtype = base_policy.dtype

    def reset(self):
        if hasattr(self.base, 'reset'):
            self.base.reset()

    def predict_action(self, obs_dict):
        # B1: fault_spec=None -> delegate to the ORIGINAL, completely
        # unmodified predict_action. Zero risk of any subtle reimplementation
        # bug affecting the B1 baseline -- only FAVOR conditions (fault_spec
        # set) exercise the custom logic below.
        if self.fault_spec is None:
            return self.base.predict_action(obs_dict)

        base = self.base
        nobs = base.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = base.horizon
        Da = base.action_dim
        Do = base.obs_feature_dim if hasattr(base, 'obs_feature_dim') else None

        this_nobs = dict_apply(nobs, lambda x: x[:, :base.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = base.obs_encoder(this_nobs)
        if base.obs_as_global_cond:
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            raise NotImplementedError("obs_as_global_cond=False not used by this project's configs")

        # self.kwargs is confirmed empty for this project's configs (every
        # __init__ parameter is explicitly named in policy: config, verified
        # this session) -- safe to omit **base.kwargs from the calls below.
        if self.mode == 'eci':
            nsample = eci_conditional_sample(
                base, cond_data, cond_mask, fault_spec=self.fault_spec,
                global_cond=global_cond)
        else:  # posthoc
            nsample = base.conditional_sample(
                cond_data, cond_mask, global_cond=global_cond)
            nsample = nsample.clone()
            nsample[..., :7] = project_fault(nsample[..., :7], self.fault_spec)

        # Matches original predict_action EXACTLY: slice action dims BEFORE
        # unnormalizing (not after) -- verified against source line 264-265.
        Da = base.action_dim
        naction_pred = nsample[..., :Da]
        action_pred = base.normalizer['action'].unnormalize(naction_pred)
        start = base.n_obs_steps - 1
        end = start + base.n_action_steps
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}
