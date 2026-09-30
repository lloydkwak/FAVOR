"""PrioIKPolicy: sample exactly as B1 (same seed stream), then task-priority IK
(ik_priority.py) on the executed waypoints. Separate file: baselines untouched."""
import torch
import sys
sys.path.insert(0, '/workspace/diffusion_policy')
sys.path.insert(0, '/workspace/docker')
from diffusion_policy.common.pytorch_util import dict_apply
from native_joint_policy import NativeJointPolicy, JOINT_NAME_TO_IDX
from fault_kinematics import PandaKinematics
from ik_priority import ik_priority


class PrioIKPolicy(NativeJointPolicy):
    def __init__(self, base_policy, lam2=0.1, **kw):
        kw['mode'] = 'eci'
        super().__init__(base_policy, **kw)
        assert self.fault_type in (None, 'locked', 'range_reduced'), self.fault_type
        self.mode, self.lam2, self.prio_log = 'prio', lam2, []

    def reset(self):
        super().reset()
        if self._kin is None:
            self._kin = PandaKinematics(device=str(self.device))
        if self._base_pos is not None and self._base_rot is not None:
            base_pos, base_rot = self._base_pos, self._base_rot
        else:
            base_pos, base_rot = self.env_ref.call('get_base_pose')[0]
        self._kin.set_base_transform(torch.as_tensor(base_pos, dtype=torch.float32, device=self.device),
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
        nsample = base.conditional_sample(cond_data, cond_mask, global_cond=global_cond, generator=generator)
        action_pred = base.normalizer['action'].unnormalize(nsample[..., :Da])
        if fault_spec is not None:
            action_pred = action_pred.clone()
            q_lo, q_hi = fault_spec['q_lo'], fault_spec['q_hi']
            if q_lo.dim() == 1:
                q_lo, q_hi = q_lo.unsqueeze(0).expand(B, 7), q_hi.unsqueeze(0).expand(B, 7)
            j = JOINT_NAME_TO_IDX[self.fault_joint_name]
            s0 = base.n_obs_steps - 1; s1 = s0 + base.n_action_steps
            q_exec = action_pred[:, s0:s1, :7]; K = q_exec.shape[1]
            lo_j = q_lo[:, j].to(q_exec).view(B, 1).expand(B, K)
            hi_j = q_hi[:, j].to(q_exec).view(B, 1).expand(B, K)
            q_con = torch.max(torch.min(q_exec[..., j], hi_j), lo_j)
            need = ((q_con - q_exec[..., j]).abs() > 1e-6).reshape(-1)
            if need.any():
                flat = q_exec.reshape(B * K, 7).clone(); idx = need.nonzero().squeeze(-1)
                q_new, info = ik_priority(self._kin, flat[idx], j, q_con.reshape(-1)[idx], lam2=self.lam2)
                flat[idx] = q_new.to(flat.dtype)
                action_pred[:, s0:s1, :7] = flat.reshape(B, K, 7)
                self.prio_log.append({'pos_after': info['pos_err_after'].mean().item(),
                                      'rot_after': info['rot_err_after'].mean().item(),
                                      'dq': info['dq_free'].mean().item()})
        start = base.n_obs_steps - 1
        end = start + base.n_action_steps
        return {'action': action_pred[:, start:end], 'action_pred': action_pred}


def summarize_prio_log(policy):
    log = getattr(policy, 'prio_log', None)
    if not log:
        return None
    out = {k: float(sum(d[k] for d in log) / len(log)) for k in log[0]}
    out['n_calls'] = len(log)
    return out
