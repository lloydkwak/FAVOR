"""RGNativeJointPolicy: reachability-guided sampling in the healthy chart
(reach_guided.py) + faulty-robot IK at the execution boundary (= B-IK, pose
weights by default). Separate file: baselines' modules are untouched."""
import torch
import sys
sys.path.insert(0, '/workspace/diffusion_policy')
sys.path.insert(0, '/workspace/docker')
from diffusion_policy.common.pytorch_util import dict_apply
from native_joint_policy import NativeJointPolicy, JOINT_NAME_TO_IDX
from fault_kinematics import PandaKinematics
from ik_redistribution import ik_redistribute
from reach_guided import rg_ddpm_sample, RGConfig


class RGNativeJointPolicy(NativeJointPolicy):
    def __init__(self, base_policy, rg_cfg=None, exec_ik=None, **kw):
        kw['mode'] = 'eci'                 # passes parent validation; switched below
        super().__init__(base_policy, **kw)
        assert self.fault_type in (None, 'locked', 'range_reduced'), self.fault_type
        self.mode = 'rg'
        self.rg_cfg = rg_cfg or RGConfig()
        self.exec_ik = {'rot_weight': 1.0} if exec_ik is None else dict(exec_ik)
        self.rg_log = []

    def reset(self):
        super().reset()
        if self._kin is None:
            self._kin = PandaKinematics(device=str(self.device))
        if self._base_pos is not None and self._base_rot is not None:
            base_pos, base_rot = self._base_pos, self._base_rot
        else:
            base_pos, base_rot = self.env_ref.call('get_base_pose')[0]
        self._kin.set_base_transform(
            torch.as_tensor(base_pos, dtype=torch.float32, device=self.device),
            torch.as_tensor(base_rot, dtype=torch.float32, device=self.device))

    def predict_action(self, obs_dict):
        base = self.base
        nobs = base.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T, Da = base.horizon, base.action_dim
        this_nobs = dict_apply(nobs, lambda x: x[:, :base.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        global_cond = base.obs_encoder(this_nobs).reshape(B, -1)
        cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        generator = self._next_generator()
        fault_spec = self._get_fault_spec()
        if fault_spec is None:
            nsample = base.conditional_sample(cond_data, cond_mask, global_cond=global_cond,
                                              generator=generator)
            action_pred = base.normalizer['action'].unnormalize(nsample[..., :Da])
        else:
            q_lo, q_hi = fault_spec['q_lo'], fault_spec['q_hi']
            if q_lo.dim() == 1:
                q_lo, q_hi = q_lo.unsqueeze(0).expand(B, 7), q_hi.unsqueeze(0).expand(B, 7)
            q_lo, q_hi = q_lo.to(self.device), q_hi.to(self.device)
            j = JOINT_NAME_TO_IDX[self.fault_joint_name]
            st = {}
            nsample = rg_ddpm_sample(base, cond_data, cond_mask, self._kin, j, q_lo, q_hi,
                                     global_cond=global_cond, generator=generator,
                                     cfg=self.rg_cfg, stats=st)
            action_pred = base.normalizer['action'].unnormalize(nsample[..., :Da]).clone()
            s0 = base.n_obs_steps - 1
            s1 = s0 + base.n_action_steps
            q_exec = action_pred[:, s0:s1, :7]
            K = q_exec.shape[1]
            lo_j = q_lo[:, j].to(q_exec).view(B, 1).expand(B, K)
            hi_j = q_hi[:, j].to(q_exec).view(B, 1).expand(B, K)
            q_con = torch.max(torch.min(q_exec[..., j], hi_j), lo_j)
            need = ((q_con - q_exec[..., j]).abs() > 1e-6).reshape(-1)
            if need.any():
                flat = q_exec.reshape(B * K, 7).clone()
                idx = need.nonzero().squeeze(-1)
                q_new, info = ik_redistribute(self._kin, flat[idx], j, q_con.reshape(-1)[idx], **self.exec_ik)
                flat[idx] = q_new.to(flat.dtype)
                action_pred[:, s0:s1, :7] = flat.reshape(B, K, 7)
                st['exec_pos_after'] = info['pos_err_after'].mean().item()
                st['exec_dq'] = info['dq_free'].mean().item()
            self.rg_log.append(st)
        start = base.n_obs_steps - 1
        end = start + base.n_action_steps
        return {'action': action_pred[:, start:end], 'action_pred': action_pred}


def summarize_rg_log(policy):
    log = getattr(policy, 'rg_log', None)
    if not log:
        return None
    keys = set().union(*[d.keys() for d in log])
    out = {k: float(sum(d[k] for d in log if k in d) / max(1, sum(1 for d in log if k in d))) for k in keys}
    out['n_calls'] = len(log)      # (no dict | dict: container runs Python 3.8)
    return out
