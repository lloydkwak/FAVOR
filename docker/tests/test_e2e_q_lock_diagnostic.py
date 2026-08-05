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
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_e2e")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

q_lo_full = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi_full = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)
fault_spec = {'joint_idx': 3, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo_full, 'q_hi': q_hi_full}

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_e2e_diag",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint4", fault_type="locked", fault_severity=None,
    n_train=0, n_test=2, n_envs=2,
    max_steps=40,  # short episode, we just want a few predict_action calls
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
)

projector = ProjectWaypoints(K=64, iters=5)
favor = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, projector=projector, env_ref=runner.env)

# monkeypatch conditional_sample to print the RPC'd q_lock on every call (once is enough)
orig_conditional_sample = favor.conditional_sample
call_count = [0]
def traced_conditional_sample(*args, **kwargs):
    if favor.env_ref is not None:
        infos = favor.env_ref.call('get_fault_info')
        print(f"[call {call_count[0]}] live q_lock pulled via RPC per env:", [i['q_lock'] for i in infos])
    call_count[0] += 1
    return orig_conditional_sample(*args, **kwargs)
favor.conditional_sample = traced_conditional_sample

log = runner.run(favor)
print("score:", log.get("test/mean_score"))
