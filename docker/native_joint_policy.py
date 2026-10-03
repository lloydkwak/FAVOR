"""
Joint-space policy wrapper for the baselines and the sample-level interventions:
  fault_spec=None  B1: plain sampling, no correction (the environment enforces the fault)
  mode='posthoc'   clip the faulted joint after sampling (identical to B1 for locked faults)
  mode='eci'       E-C-I: project the faulted joint onto its admissible range at every denoising step
  mode='random_n'  draw N candidates, keep one at random (control for 'select')
  mode='select'    draw N candidates, keep the one with the smallest EE discrepancy (fault_certificate.py)
  mode='ik'        W-IK: weighted damped-least-squares IK on the executed waypoints (ik_redistribution.py)

Deterministic seeding: every predict_action call is seeded from (episode, call) counters, so all
methods draw identical noise and episodes can be compared pairwise across methods.

Fault spec: with env_ref + fault_joint_name/type/severity, the actual per-episode bounds are read from
the environment after each reset (FaultInjector.get_fault_info()), because the lock angle and the
reduced range depend on the joint position at reset. Healthy joints get their full physical range,
the faulted joint the locked / narrowed bounds (same formula as fault_injector.py).
"""
import torch
import sys
sys.path.insert(0, '/workspace/diffusion_policy')
sys.path.insert(0, '/workspace/docker')
from diffusion_policy.common.pytorch_util import dict_apply
from joint_eci_projector import eci_conditional_sample, project_fault, normalize_joint_bounds
from fault_kinematics import PandaKinematics
from fault_certificate import FeasibilityCertificate
from ik_redistribution import ik_redistribute

PANDA_Q_LO = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
PANDA_Q_HI = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
PANDA_QVEL_MAX = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=torch.float32)
CONTROL_DT = 1.0 / 20.0  # control_freq=20Hz (dataset env_meta)
JOINT_NAME_TO_IDX = {f"robot0_joint{i}": i - 1 for i in range(1, 8)}


class NativeJointPolicy:
    def __init__(self, base_policy, fault_spec=None, mode='eci', base_seed=0,
                 env_ref=None, fault_joint_name=None, fault_type=None, fault_severity=None,
                 n_resample=1, n_select=16, select_beta=0.05,
                 base_pos=None, base_rot=None, chunk_size=8,
                 ik_overrides=None):
        """
        fault_spec: static dict (q_lo, q_hi, [v_max, q_anchor], all (7,)) --
                    for the nominal/vacuous tests. Mutually exclusive with
                    the env-driven args below.
        env_ref/fault_joint_name/fault_type/fault_severity: if given (and
                    fault_spec is None), the ACTUAL per-episode fault_spec
                    is built dynamically from the environment after each
                    reset(). Used for the real fault sweep.
        """
        assert mode in ('eci', 'posthoc', 'select', 'random_n', 'ik')
        if mode == 'ik':
            assert fault_type in (None, 'locked', 'range_reduced'), fault_type
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

        # 'select' config. n_select: candidates per env per
        # predict_action call (D2 used N=64 for offline diversity checks;
        # 16 is a runtime-cost compromise for closed-loop rollout -- see
        # design doc section 9 risk table on the N-vs-cost tradeoff).
        self.n_select = n_select
        self._kin = None  # lazily built on first use, and base transform
        # set once per episode in reset() once env_ref is available (locked
        # joint's own base pose doesn't change within an episode).
        self._cert = None
        self.select_beta = select_beta
        self._base_pos = base_pos
        self._base_rot = base_rot

        # chunk_size: N candidates are drawn in sequential chunks of this
        # size rather than one batch of size B*N, to fit GPU memory without
        # losing candidates -- N=8 in one shot was found to lose the
        # achievable minimum epsilon by 1.7-2.2x relative to N=64 (measured
        # directly), and larger
        # single-shot batches (N=32+) hit CUDA OOM. Chunking costs wall-clock
        # time (each chunk is a full denoising run) but no accuracy, unlike
        # fp16 (tried and reverted -- a dtype mismatch surfaced elsewhere in
        # the pipeline, and half precision's effect on diffusion sampling
        # quality wasn't verified anyway).
        self.chunk_size = chunk_size

        # random_n: draws n_select candidates like select does, but picks
        # one uniformly at random instead of scoring with the certificate.
        # This is the required control for select's result to mean
        # anything: if select's improvement over B1 is no better than
        # random_n's, "N samples chosen by epsilon" isn't doing real work
        # beyond "N samples, pick any" -- the whole point of the certificate
        # is to beat this, not just to beat single-sample B1.
        self._random_n_gen = torch.Generator()

        # 'ik' (W-IK baseline): see ik_redistribution.py
        self.ik_overrides = dict(ik_overrides or {})  # empty -> ik_redistribution module defaults
        self.ik_log = []

    def reset(self):
        if hasattr(self.base, 'reset'):
            self.base.reset()
        self._episode_idx += 1
        self._call_idx = 0
        self._dynamic_fault_spec = None  # force rebuild on next predict_action

        if self.mode in ('select', 'ik'):
            if self._kin is None:
                self._kin = PandaKinematics(device=str(self.device))
                self._cert = FeasibilityCertificate(self._kin, beta=self.select_beta)
            # base pose read once per episode via FaultInjector.get_base_pose()
            # (added alongside get_current_qpos/get_fault_info's existing
            # RPC pattern) -- NOT cached across episodes, see that method's
            # docstring on why.
            if self._base_pos is not None and self._base_rot is not None:
                base_pos, base_rot = self._base_pos, self._base_rot
            else:
                assert self.env_ref is not None, (
                    "select/ik needs either explicit base_pos/base_rot "
                    "or env_ref to query them via get_base_pose()")
                poses = self.env_ref.call('get_base_pose')
                base_pos, base_rot = poses[0]  # identical across envs (same robot mount)
            self._kin.set_base_transform(
                torch.as_tensor(base_pos, dtype=torch.float32, device=self.device),
                torch.as_tensor(base_rot, dtype=torch.float32, device=self.device))

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

    def _draw_n_candidates_chunked(self, cond_data, cond_mask, global_cond, N):
        """Draws N candidates per env by running conditional_sample in
        sequential chunks of self.chunk_size (each chunk still batches
        B*chunk_size internally), concatenating results along a new
        candidate dimension. See constructor docstring on chunk_size for
        why this exists instead of one B*N batch.

        Each chunk gets its own generator, advanced from self._call_idx
        exactly like _next_generator -- otherwise repeating the same
        generator across chunks would draw IDENTICAL noise per chunk,
        silently duplicating candidates instead of adding real diversity.
        """
        B = cond_data.shape[0]
        chunks = []
        remaining = N
        while remaining > 0:
            n_this = min(self.chunk_size, remaining)
            gen = self._next_generator()
            cond_data_rep = cond_data.repeat_interleave(n_this, dim=0)
            cond_mask_rep = cond_mask.repeat_interleave(n_this, dim=0)
            global_cond_rep = global_cond.repeat_interleave(n_this, dim=0)
            nsample = self.base.conditional_sample(
                cond_data_rep, cond_mask_rep, global_cond=global_cond_rep, generator=gen)
            chunks.append(nsample.reshape(B, n_this, *nsample.shape[1:]))
            remaining -= n_this
        return torch.cat(chunks, dim=1)  # (B, N, T, Da)

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

        if self.mode == 'random_n' and fault_spec is not None:
            # Same N-candidate draw as select, but no certificate scoring:
            # picks one candidate per env uniformly at random. Seeded
            # deterministically from (episode, call) exactly like
            # _next_generator, so repeated runs with the same base_seed
            # reproduce the same random choice -- required for this to be
            # a valid paired control against select/B1 under identical
            # seeds, not just "some random baseline that happens to run".
            N = self.n_select
            nsample_all = self._draw_n_candidates_chunked(cond_data, cond_mask, global_cond, N)  # (B, N, T, Da)
            naction_pred_all = nsample_all[..., :Da]
            action_pred_all = base.normalizer['action'].unnormalize(
                naction_pred_all.reshape(B * N, T, Da)).reshape(B, N, T, Da)

            # NOTE: self._call_idx was already incremented by the
            # _next_generator() call above (which supplied `generator` for
            # the diffusion sampling itself); using it here means this
            # random-index draw is seeded one step ahead of that call's own
            # seed, not off it. That's fine for random_n's OWN
            # reproducibility (still fully deterministic run-to-run), but
            # it does mean this seed does not equal `generator`'s -- by
            # design, since a single shared seed would correlate the
            # diffusion noise and the index draw for no reason. Recorded
            # here explicitly since it's easy to assume otherwise.
            rn_seed = self.base_seed * 1_000_000 + self._episode_idx * 1000 + self._call_idx
            self._random_n_gen.manual_seed(rn_seed)
            choice = torch.randint(0, N, (B,), generator=self._random_n_gen)
            action_pred = action_pred_all[torch.arange(B), choice]

            start = base.n_obs_steps - 1
            end = start + base.n_action_steps
            action = action_pred[:, start:end]
            return {'action': action, 'action_pred': action_pred}

        if self.mode == 'select' and fault_spec is not None:
            # Draw n_select candidates PER ENV in one batched diffusion
            # call (expand B -> B*n_select along the batch dim, since the
            # denoiser has no cross-sample dependence), score each by
            # epsilon(Q) (fault_certificate.py -- EE discrepancy the fault
            # would introduce, NOT a joint-space penalty on the faulted
            # joint itself: see that module's docstring on why the
            # faulted joint alone is not the right thing to score), and
            # keep the argmin per env.
            N = self.n_select
            nsample_all = self._draw_n_candidates_chunked(cond_data, cond_mask, global_cond, N)  # (B, N, T, Da)
            naction_pred_all = nsample_all[..., :Da]
            action_pred_all = base.normalizer['action'].unnormalize(
                naction_pred_all.reshape(B * N, T, Da)).reshape(B, N, T, Da)

            joint_name = self.fault_joint_name
            j_idx = JOINT_NAME_TO_IDX[joint_name]
            q_lo_b, q_hi_b = fault_spec['q_lo'][:, j_idx], fault_spec['q_hi'][:, j_idx]  # (B,)
            v_max_b = fault_spec.get('v_max')
            q_anchor_b = fault_spec.get('q_anchor')

            action_pred = torch.zeros(B, T, Da, device=self.device, dtype=self.dtype)
            for b in range(B):
                Q = action_pred_all[b, :, :, :7]  # (N, T, 7)
                if self.fault_type == 'locked':
                    params = {'q_lock': q_lo_b[b]}  # q_lo==q_hi for locked
                elif self.fault_type == 'range_reduced':
                    params = {'q_lo': q_lo_b[b], 'q_hi': q_hi_b[b]}
                elif self.fault_type == 'velocity_limited':
                    q_prev_b = q_anchor_b[b, j_idx].expand(N, 1).to(Q.device)
                    q_prev_seq = torch.cat(
                        [q_prev_b, Q[:, :-1, j_idx]], dim=1) if T > 1 else q_prev_b
                    params = {'v_max': v_max_b[b, j_idx], 'q_prev': q_prev_seq}
                else:
                    raise ValueError(f"unsupported fault_type for select mode: {self.fault_type!r}")

                eps, _ = self._cert.score(Q, self.fault_type, joint_name, params)
                best = torch.argmin(eps).item()
                action_pred[b] = action_pred_all[b, best]

            start = base.n_obs_steps - 1
            end = start + base.n_action_steps
            action = action_pred[:, start:end]
            return {'action': action, 'action_pred': action_pred}

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

        if fault_spec is not None and self.mode == 'ik':
            # W-IK: same sample as B1 (this branch sampled exactly like B1
            # above), then keep the intended EE pose (FK of the targets), fix
            # the faulted joint to its admissible value and re-solve the 6
            # healthy joints by damped least-squares IK. Only the waypoints
            # that will be executed are touched; already-admissible ones stay
            # bit-identical to B1.
            action_pred = action_pred.clone()
            s0 = base.n_obs_steps - 1
            s1 = s0 + base.n_action_steps
            j = JOINT_NAME_TO_IDX[self.fault_joint_name]
            q_exec = action_pred[:, s0:s1, :7]
            K = q_exec.shape[1]
            lo_j = fault_spec['q_lo'][:, j].to(q_exec.device, q_exec.dtype).view(B, 1).expand(B, K)
            hi_j = fault_spec['q_hi'][:, j].to(q_exec.device, q_exec.dtype).view(B, 1).expand(B, K)
            q_con = torch.max(torch.min(q_exec[..., j], hi_j), lo_j)
            need = ((q_con - q_exec[..., j]).abs() > 1e-6).reshape(-1)
            if need.any():
                flat = q_exec.reshape(B * K, 7).clone()
                idx = need.nonzero().squeeze(-1)
                q_new, info = ik_redistribute(
                    self._kin, flat[idx], j, q_con.reshape(-1)[idx], **self.ik_overrides)
                flat[idx] = q_new.to(flat.dtype)
                action_pred[:, s0:s1, :7] = flat.reshape(B, K, 7)
                self.ik_log.append({
                    'n': int(idx.numel()),
                    'pos_before': info['pos_err_before'].mean().item(),
                    'pos_after': info['pos_err_after'].mean().item(),
                    'rot_before': info['rot_err_before'].mean().item(),
                    'rot_after': info['rot_err_after'].mean().item(),
                    'dq': info['dq_free'].mean().item()})

        start = base.n_obs_steps - 1
        end = start + base.n_action_steps
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}
