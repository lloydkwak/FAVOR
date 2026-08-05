"""
Regression test: FavorHybridImagePolicy must pull each environment's OWN
live q_lock via env_ref.call('get_fault_info') RPC -- not a hardcoded value.
This is the exact bug class found when q_lock was hardcoded to 0.0 while the
real environment lock value was ~-2.6 (Lift/joint4).
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from ik_projector import ProjectWaypoints
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_env_ref_test")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
fault_spec = {'joint_idx': 3, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_scratch_env_ref_test_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint4", fault_type="locked", fault_severity=None,
    n_train=0, n_test=2, n_envs=2,
    max_steps=16, n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
)
projector = ProjectWaypoints(K=64, iters=5)
favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, projector=projector, env_ref=runner.env)

pulled = []
orig_conditional_sample = favor.conditional_sample
def traced(*args, **kwargs):
    infos = favor.env_ref.call('get_fault_info')
    pulled.append([i['q_lock'] for i in infos])
    return orig_conditional_sample(*args, **kwargs)
favor.conditional_sample = traced

runner.run(favor)

assert len(pulled) > 0, "conditional_sample was never called"
first = pulled[0]
assert len(first) == 2, f"expected 2 per-env values, got {len(first)}"
assert abs(first[0]) > 0.5 and abs(first[1]) > 0.5, \
    f"pulled q_lock values look like the 0.0 placeholder, not a real reset() value: {first}"
assert first[0] != first[1], f"two envs got identical q_lock ({first}) -- expected different random resets"
for call in pulled:
    assert call == first, "q_lock changed mid-episode -- should be constant (locked = fixed at onset)"

print("pulled per-env q_lock (first call):", first)
print("ENV_REF_WIRING_VERIFIED")
