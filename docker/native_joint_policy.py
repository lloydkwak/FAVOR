"""
Native joint-space policy wrapper for actuation_mode='joint' evaluation,
used by both B1 (no correction) and FAVOR (E-C-I / post-hoc projection)
conditions. Replaces the EE-pose-era FavorHybridImagePolicy for checkpoints
retrained natively in joint space (no IK anywhere in this module).

DETERMINISTIC SEEDING: diffusion sampling is stochastic. This class seeds
every predict_action call deterministically from (episode counter, call
counter) so B1 and FAVOR draw identical noise whenever their algorithms
are identical (see this session's identity-check findings).

DYNAMIC ENV-DRIVEN FAULT SPEC (added for the fault sweep): rather than a
static fault_spec passed at construction, FAVOR conditions can be given
env_ref + fault_joint_name + fault_type + fault_severity, and will query
the ACTUAL fault parameters from the environment (FaultInjector.get_fault_info())
once per episode (right after reset, cached until the next reset). This
mirrors the EE-pose-era FavorHybridImagePolicy's pattern and is necessary
because the locked value (q_lock) and the range_reduced bounds depend on
the joint's position AT RESET TIME (self._q_onset in fault_injector.py),
which cannot be known before the episode starts. All 7 joints get a
q_lo/q_hi entry -- healthy joints get their full physical range (identity
projection), the faulted joint gets the actual narrowed/locked/rate-limited
bounds, EXACTLY mirroring fault_injector.py's own range_reduced formula
(verified against source this session: new_lo/hi = clip(q_onset -+ 0.5*
severity*(hi-lo), lo, hi)) and velocity limit (PANDA_QVEL_MAX * severity,
converted from rad/s to rad/waypoint using the environment's control_freq=20Hz,
i.e. dt=0.05s per waypoint -- confirmed via the dataset's env_meta this
session; also confirmed each predicted trajectory waypoint corresponds to
exactly one env.step() call).
"""
import torch
import sys
sys.path.insert(0, '/workspace/diffusion_policy')
from diffusion_policy.common.pytorch_util import dict_apply
from joint_eci_projector import eci_conditional_sample, project_fault, normalize_joint_bounds

PANDA_Q_LO = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
PANDA_Q_HI = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
PANDA_QVEL_MAX = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=torch.float32)
CONTROL_DT = 1.0 / 20.0  # control_freq=20Hz, confirmed from dataset env_meta this session
JOINT_NAME_TO_IDX = {f"robot0_joint{i}": i - 1 for i in range(1, 8)}


class NativeJointPolicy:
    def __init__(self, base_policy, fault_spec=None, mode='eci', base_seed=0,
                 env_ref=None, fault_joint_name=None, fault_type=None, fault_severity=None,
                 n_resample=1):
        """
        fault_spec: static dict (q_lo, q_hi, [v_max, q_anchor], all (7,)) --
                    for the nominal/vacuous tests. Mutually exclusive with
                    the env-driven args below.
        env_ref/fault_joint_name/fault_type/fault_severity: if given (and
                    fault_spec is None), the ACTUAL per-episode fault_spec
                    is built dynamically from the environment after each
                    reset(). Used for the real fault sweep.
        """
        assert mode in ('eci', 'posthoc')
        self.base = base_policy
        self.static_fault_spec = fault_spec
        self.mode = mode
        self.device = base_policy.device
        self.dtype = base_policy.dtype
        self.base_seed = base_seed
        self._episode_idx = -1
        self._call_idx = 0

        self.env_ref = env_ref
        self.fault_joint_name = fault_joint_name
        self.fault_type = fault_type
        self.fault_severity = fault_severity
        self._dynamic_fault_spec = None  # rebuilt each reset() if env_ref given
        self.n_resample = n_resample

    def reset(self):
        if hasattr(self.base, 'reset'):
            self.base.reset()
        self._episode_idx += 1
        self._call_idx = 0
        self._dynamic_fault_spec = None  # force rebuild on next predict_action

    def _next_generator(self):
        seed = self.base_seed * 1_000_000 + self._episode_idx * 1000 + self._call_idx
        self._call_idx += 1
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        return gen

    def _build_dynamic_fault_spec(self):
        """
        Queries the ACTUAL per-env fault state and builds a (B,7) q_lo/q_hi
        (and v_max/q_anchor if velocity_limited) fault_spec, exactly
        mirroring fault_injector.py's own physics.
        """
        infos = self.env_ref.call('get_fault_info')
        qpos_list = self.env_ref.call('get_current_qpos')
        B = len(infos)
        q_lo = PANDA_Q_LO.unsqueeze(0).expand(B, 7).clone()
        q_hi = PANDA_Q_HI.unsqueeze(0).expand(B, 7).clone()
        q_anchor = torch.tensor([list(q) for q in qpos_list], dtype=torch.float32)
        v_max = torch.full((B, 7), 1e6)  # effectively unconstrained by default

        for b, info in enumerate(infos):
            ftype = info['fault_type']
            if ftype is None:
                continue
            jidx = JOINT_NAME_TO_IDX[self.fault_joint_name]
            q_onset = info['q_lock']
            if ftype == 'locked':
                q_lo[b, jidx] = q_onset
                q_hi[b, jidx] = q_onset
            elif ftype == 'range_reduced':
                lo_phys, hi_phys = PANDA_Q_LO[jidx].item(), PANDA_Q_HI[jidx].item()
                s = hi_phys - lo_phys
                half = 0.5 * s * info['severity']
                new_lo = max(min(q_onset - half, hi_phys), lo_phys)
                new_hi = max(min(q_onset + half, hi_phys), lo_phys)
                q_lo[b, jidx] = new_lo
                q_hi[b, jidx] = new_hi
            elif ftype == 'velocity_limited':
                qmax_rad_s = PANDA_QVEL_MAX[jidx].item() * info['severity']
                v_max[b, jidx] = qmax_rad_s * CONTROL_DT  # rad/s -> rad/waypoint

        return {'q_lo': q_lo, 'q_hi': q_hi, 'v_max': v_max, 'q_anchor': q_anchor}

    def _get_fault_spec(self):
        if self.static_fault_spec is not None:
            return self.static_fault_spec
        if self.env_ref is None:
            return None
        if self._dynamic_fault_spec is None:
            self._dynamic_fault_spec = self._build_dynamic_fault_spec()
        return self._dynamic_fault_spec

    def predict_action(self, obs_dict):
        base = self.base
        nobs = base.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = base.horizon
        Da = base.action_dim
        this_nobs = dict_apply(nobs, lambda x: x[:, :base.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = base.obs_encoder(this_nobs)
        if base.obs_as_global_cond:
            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            raise NotImplementedError("obs_as_global_cond=False not used by this project's configs")

        generator = self._next_generator()
        fault_spec = self._get_fault_spec()

        if fault_spec is None:
            # B1: no projection at all. The environment itself still
            # physically enforces the fault (if any) -- this policy branch
            # is intentionally identical whether or not the env has a
            # fault, matching B1's definition (fault-blind).
            nsample = base.conditional_sample(
                cond_data, cond_mask, global_cond=global_cond, generator=generator)
        elif self.mode == 'eci':
            q_lo_n, q_hi_n, v_max_n, q_anchor_n = normalize_joint_bounds(
                base, fault_spec['q_lo'], fault_spec['q_hi'],
                v_max_phys=fault_spec.get('v_max'), q_anchor_phys=fault_spec.get('q_anchor'))
            normalized_fault_spec = {'q_lo': q_lo_n, 'q_hi': q_hi_n}
            if v_max_n is not None:
                normalized_fault_spec['v_max'] = v_max_n
                normalized_fault_spec['q_anchor'] = q_anchor_n
            nsample = eci_conditional_sample(
                base, cond_data, cond_mask, fault_spec=normalized_fault_spec,
                global_cond=global_cond, generator=generator, n_resample=self.n_resample)
        else:  # posthoc
            nsample = base.conditional_sample(
                cond_data, cond_mask, global_cond=global_cond, generator=generator)

        naction_pred = nsample[..., :Da]
        action_pred = base.normalizer['action'].unnormalize(naction_pred)

        if fault_spec is not None and self.mode == 'posthoc':
            action_pred = action_pred.clone()
            action_pred[..., :7] = project_fault(
                action_pred[..., :7], fault_spec['q_lo'], fault_spec['q_hi'],
                v_max=fault_spec.get('v_max'), q_anchor=fault_spec.get('q_anchor'))

        start = base.n_obs_steps - 1
        end = start + base.n_action_steps
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}
