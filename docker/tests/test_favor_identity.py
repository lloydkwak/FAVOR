import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_4_1")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

favor_policy = FavorHybridImagePolicy(base_policy, fault_spec=None)

shape_meta = cfg.task.shape_meta
B = 4
n_obs_steps = cfg.task.env_runner.n_obs_steps
obs_dict = {}
for key, attr in shape_meta['obs'].items():
    shape = tuple(attr['shape'])
    obs_dict[key] = torch.randn(B, n_obs_steps, *shape, device="cuda:0")

FAILURES = 0
for trial in range(3):
    seed = 1000 + trial
    torch.manual_seed(seed)
    with torch.no_grad():
        out_b1 = base_policy.predict_action(obs_dict)
    torch.manual_seed(seed)
    with torch.no_grad():
        out_favor = favor_policy.predict_action(obs_dict)

    identical = torch.equal(out_b1['action'], out_favor['action'])
    max_diff = (out_b1['action'] - out_favor['action']).abs().max().item()
    print(trial, identical, max_diff)
    if not identical:
        FAILURES += 1

print("FAILURES", FAILURES)
if FAILURES:
    sys.exit(1)
print("ALL_IDENTICAL")
