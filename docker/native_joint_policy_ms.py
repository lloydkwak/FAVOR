"""MisspecPrioPolicy: Priority IK under fault-knowledge misspecification.
ms_mode:
  range_scale (param s)  : believed window = true window scaled by s about its center
  lock_offset (param d)  : believed lock angle = true + d (rad)
  wrong_joint            : believed faulty joint = neighbour (j+1, or j-1 for j7), window of
                           the true width centred on that joint's measured angle at onset
  online                 : no prior knowledge. Detect the faulty joint from command-vs-measured
                           tracking error (> det_tau for det_k consecutive chunks), estimate the
                           window from the observed min/max of that joint (+margin), then apply
                           Priority IK. Before detection = B1 (identical sampling).
Separate file: baselines and PrioIKPolicy untouched."""
import sys
sys.path.insert(0, '/workspace/diffusion_policy')
sys.path.insert(0, '/workspace/docker')
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from native_joint_policy import JOINT_NAME_TO_IDX, PANDA_Q_LO, PANDA_Q_HI
import native_joint_policy_prio as npp
from native_joint_policy_prio import PrioIKPolicy

IDX_TO_NAME = {v: k for k, v in JOINT_NAME_TO_IDX.items()}


def _clip(x, lo, hi):
    return torch.max(torch.min(x, hi), lo)


class MisspecPrioPolicy(PrioIKPolicy):
    def __init__(self, base_policy, ms_mode, ms_param=None, lam2=0.2,
                 det_tau=0.08, det_k=2, det_margin=0.01, det_rel=0.0,
                 det_mode='track', still_eps=1e-3, cmd_eps=0.01, **kw):
        super().__init__(base_policy, lam2=lam2, **kw)
        assert ms_mode in ('range_scale', 'lock_offset', 'wrong_joint', 'online', 'delay'), ms_mode
        self._ncalls = 0
        self.ms_mode, self.ms_param = ms_mode, ms_param
        self.det_tau, self.det_k, self.det_margin, self.det_rel = det_tau, det_k, det_margin, det_rel
        self.healthy_err = []   # tracking errors of truly healthy joints (for calibration)
        self.det_mode, self.still_eps, self.cmd_eps = det_mode, still_eps, cmd_eps
        self._prev_meas = None
        self._true_joint = self.fault_joint_name
        if ms_mode == 'wrong_joint':
            j = JOINT_NAME_TO_IDX[self._true_joint]
            self.fault_joint_name = IDX_TO_NAME[j + 1 if j < 6 else j - 1]
        self.ms_log = []
        self._det = None
        self._last_cmd = None
        self._true_win = None

    # ---------------------------------------------------------------- reset
    def reset(self):
        super().reset()
        self._flush()
        self._det, self._last_cmd, self._true_win, self._prev_meas = None, None, None, None
        self._ncalls = 0

    def _flush(self):
        d = self._det
        if d is None or self._true_win is None:
            return
        jt = JOINT_NAME_TO_IDX[self._true_joint]
        for b in range(len(d['joint'])):
            j = d['joint'][b]
            e = {'true_j': jt, 'det_j': j, 'det_call': d['call'][b], 'n_calls': d['ncalls'],
                 'true_lo': float(self._true_win[0][b]), 'true_hi': float(self._true_win[1][b])}
            if j >= 0:
                e['est_lo'] = float(d['qmin'][b, j]) - self.det_margin
                e['est_hi'] = float(d['qmax'][b, j]) + self.det_margin
            self.ms_log.append(e)
        self._det = None

    # ------------------------------------------------------ misspecified spec
    def _build_dynamic_fault_spec(self):
        saved = self.fault_joint_name
        self.fault_joint_name = self._true_joint
        try:
            spec = super()._build_dynamic_fault_spec()
        finally:
            self.fault_joint_name = saved
        q_lo, q_hi = spec['q_lo'].clone(), spec['q_hi'].clone()
        jt = JOINT_NAME_TO_IDX[self._true_joint]
        self._true_win = (q_lo[:, jt].clone(), q_hi[:, jt].clone())
        LO, HI = PANDA_Q_LO.to(q_lo), PANDA_Q_HI.to(q_lo)
        if self.ms_mode == 'range_scale':
            c = 0.5 * (q_lo[:, jt] + q_hi[:, jt])
            h = 0.5 * (q_hi[:, jt] - q_lo[:, jt]) * float(self.ms_param)
            q_lo[:, jt] = _clip(c - h, LO[jt], HI[jt]); q_hi[:, jt] = _clip(c + h, LO[jt], HI[jt])
        elif self.ms_mode == 'lock_offset':
            v = _clip(q_lo[:, jt] + float(self.ms_param), LO[jt], HI[jt])
            q_lo[:, jt] = v; q_hi[:, jt] = v
        elif self.ms_mode == 'wrong_joint':
            k = JOINT_NAME_TO_IDX[self.fault_joint_name]
            w = q_hi[:, jt] - q_lo[:, jt]
            a = spec['q_anchor'][:, k].to(q_lo)
            q_lo[:, jt] = LO[jt]; q_hi[:, jt] = HI[jt]
            q_lo[:, k] = _clip(a - 0.5 * w, LO[k], HI[k]); q_hi[:, k] = _clip(a + 0.5 * w, LO[k], HI[k])
        out = dict(spec); out['q_lo'] = q_lo; out['q_hi'] = q_hi
        return out

    # ------------------------------------------------------------- online mode
    def _get_fault_spec(self):
        # 'delay': exact fault knowledge, but correction withheld for the first ms_param calls
        if self.ms_mode == 'delay' and self._ncalls <= int(self.ms_param):
            return None
        return super()._get_fault_spec()

    def predict_action(self, obs_dict):
        self._ncalls += 1
        if self.ms_mode != 'online':
            return super().predict_action(obs_dict)
        if self._true_win is None:
            self._get_fault_spec()          # only to record the true window for logging
        base = self.base
        nobs = base.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T, Da = base.horizon, base.action_dim
        this_nobs = dict_apply(nobs, lambda x: x[:, :base.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        global_cond = base.obs_encoder(this_nobs).reshape(B, -1)
        cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        generator = self._next_generator()
        nsample = base.conditional_sample(cond_data, cond_mask, global_cond=global_cond, generator=generator)
        action_pred = base.normalizer['action'].unnormalize(nsample[..., :Da]).clone()

        q_meas = torch.tensor([list(q) for q in self.env_ref.call('get_current_qpos')], dtype=torch.float32)
        if self._det is None:
            self._det = {'qmin': q_meas.clone(), 'qmax': q_meas.clone(),
                         'cnt': torch.zeros(B, 7, dtype=torch.long),
                         'joint': [-1] * B, 'call': [-1] * B, 'ncalls': 0}
        d = self._det
        d['ncalls'] += 1
        d['qmin'] = torch.min(d['qmin'], q_meas); d['qmax'] = torch.max(d['qmax'], q_meas)
        if self._last_cmd is not None:
            err = (self._last_cmd - q_meas).abs()
            if self.det_mode == 'stuck' and self._prev_meas is not None:
                moved = (q_meas - self._prev_meas).abs()
                want = (self._last_cmd - self._prev_meas).abs()
                over = (moved < self.still_eps) & (want > self.cmd_eps)
            else:
                over = err > self.det_tau
            if self.det_rel > 0 and self.det_mode != 'stuck':
                top2 = err.topk(2, dim=1)
                am = top2.indices[:, :1]
                other_max = torch.where(torch.arange(7).unsqueeze(0) == am,
                                        top2.values[:, 1:2], top2.values[:, :1])
                over = over & (err > self.det_rel * other_max)
            d['cnt'] = torch.where(over, d['cnt'] + 1, torch.zeros_like(d['cnt']))
            jt = JOINT_NAME_TO_IDX[self._true_joint]
            hmask = torch.ones(7, dtype=torch.bool); hmask[jt] = False
            self.healthy_err.extend(err[:, hmask].reshape(-1).tolist())
            for b in range(B):
                if d['joint'][b] < 0 and bool((d['cnt'][b] >= self.det_k).any()):
                    cand = torch.where(d['cnt'][b] >= self.det_k, err[b], torch.zeros_like(err[b]))
                    d['joint'][b] = int(cand.argmax()); d['call'][b] = d['ncalls']

        s0 = base.n_obs_steps - 1; s1 = s0 + base.n_action_steps
        for b in range(B):
            j = d['joint'][b]
            if j < 0:
                continue
            lo = float(d['qmin'][b, j]) - self.det_margin
            hi = float(d['qmax'][b, j]) + self.det_margin
            q_exec = action_pred[b, s0:s1, :7].clone()
            q_con = q_exec[:, j].clamp(lo, hi)
            need = (q_con - q_exec[:, j]).abs() > 1e-6
            if bool(need.any()):
                idx = need.nonzero().squeeze(-1)
                q_new, info = npp.ik_priority(self._kin, q_exec[idx], j, q_con[idx], lam2=self.lam2)
                q_exec[idx] = q_new.to(q_exec.dtype)
                action_pred[b, s0:s1, :7] = q_exec
        self._last_cmd = action_pred[:, s1 - 1, :7].detach().cpu().float()
        self._prev_meas = q_meas
        return {'action': action_pred[:, s0:s1], 'action_pred': action_pred}


def summarize_ms_log(policy):
    policy._flush()
    log = policy.ms_log
    if not log:
        return None
    n = len(log)
    det = [e for e in log if e['det_j'] >= 0]
    cor = [e for e in det if e['det_j'] == e['true_j']]
    out = {'n_env_episodes': n, 'detect_rate': len(det) / n,
           'correct_rate': len(cor) / n, 'wrong_joint_rate': (len(det) - len(cor)) / n}
    if det:
        out['mean_detect_call'] = sum(e['det_call'] for e in det) / len(det)
    he = sorted(getattr(policy, 'healthy_err', []))
    if he:
        out['healthy_err_p50'] = he[len(he) // 2]; out['healthy_err_p95'] = he[int(0.95 * (len(he) - 1))]
        out['healthy_err_max'] = he[-1]
    if cor:
        out['mean_lo_err'] = sum(abs(e['est_lo'] - e['true_lo']) for e in cor) / len(cor)
        out['mean_hi_err'] = sum(abs(e['est_hi'] - e['true_hi']) for e in cor) / len(cor)
    return out
