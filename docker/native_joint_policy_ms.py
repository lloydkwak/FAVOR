"""MisspecPrioPolicy: Priority IK under misspecified fault knowledge.
The environment applies the TRUE fault; the policy corrects with a WRONG fault model:
  range_scale (param s) : believed admissible window = true window scaled by s about its center
  lock_offset (param d) : believed lock angle = true lock angle + d (rad)
Sampling is identical to B1 / PrioIKPolicy (same seed stream); only the fault spec differs.
Separate file: baselines and PrioIKPolicy untouched."""
import sys
sys.path.insert(0, '/workspace/diffusion_policy')
sys.path.insert(0, '/workspace/docker')
import torch
from native_joint_policy import JOINT_NAME_TO_IDX, PANDA_Q_LO, PANDA_Q_HI
from native_joint_policy_prio import PrioIKPolicy


def _clip(x, lo, hi):
    return torch.max(torch.min(x, hi), lo)


class MisspecPrioPolicy(PrioIKPolicy):
    def __init__(self, base_policy, ms_mode, ms_param, lam2=0.2, **kw):
        super().__init__(base_policy, lam2=lam2, **kw)
        assert ms_mode in ('range_scale', 'lock_offset'), ms_mode
        self.ms_mode, self.ms_param = ms_mode, float(ms_param)

    def _build_dynamic_fault_spec(self):
        spec = super()._build_dynamic_fault_spec()
        q_lo, q_hi = spec['q_lo'].clone(), spec['q_hi'].clone()
        j = JOINT_NAME_TO_IDX[self.fault_joint_name]
        LO, HI = PANDA_Q_LO.to(q_lo), PANDA_Q_HI.to(q_lo)
        if self.ms_mode == 'range_scale':
            c = 0.5 * (q_lo[:, j] + q_hi[:, j])
            h = 0.5 * (q_hi[:, j] - q_lo[:, j]) * self.ms_param
            q_lo[:, j] = _clip(c - h, LO[j], HI[j]); q_hi[:, j] = _clip(c + h, LO[j], HI[j])
        else:  # lock_offset
            v = _clip(q_lo[:, j] + self.ms_param, LO[j], HI[j])
            q_lo[:, j] = v; q_hi[:, j] = v
        out = dict(spec); out['q_lo'] = q_lo; out['q_hi'] = q_hi
        return out


def summarize_ms_log(policy):
    return None
