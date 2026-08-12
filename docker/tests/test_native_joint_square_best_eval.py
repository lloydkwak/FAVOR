"""
Formal no-fault evaluation + video recording of the best natively
joint-space-trained Square checkpoint so far (epoch 70, test_mean_score=0.400
during training's n=10 rollout). This script re-evaluates at n=28 (project
standard) with video capture for a few episodes.
"""
import sys, os
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/outputs/2026.08.10/05.53.47_train_diffusion_unet_hybrid_square_image_joint/checkpoints/epoch=0070-test_mean_score=0.400.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/square/ph/image_abs.hdf5"

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
print("checkpoint epoch:", payload.get('epoch'))
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_native_square_best")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

class NativeJointPolicyAdapter:
    def __init__(self, base_policy):
        self.base = base_policy
        self.device = base_policy.device
        self.dtype = base_policy.dtype
    def predict_action(self, obs_dict):
        return self.base.predict_action(obs_dict)
    def reset(self):
        if hasattr(self.base, 'reset'):
            self.base.reset()

policy = NativeJointPolicyAdapter(base_policy)

os.makedirs("/workspace/results/videos_native_best", exist_ok=True)
runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/videos_native_best",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=0, n_test=28, n_test_vis=5, test_start_seed=10000, n_envs=28,
    max_steps=400, n_obs_steps=2, n_action_steps=8,
    render_obs_key='agentview_image',
    fps=10, crf=22,
    abs_action=True,
    actuation_mode='joint',
)
log = runner.run(policy)
print("Square, native joint-space (epoch 70, best so far), no fault, n=28: score =", log.get("test/mean_score"))
for k, v in log.items():
    if 'sim_video' in k:
        print(f"  video: {k}")
