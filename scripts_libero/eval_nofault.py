"""
No-fault B1 evaluation of an existing checkpoint (no retraining).
Usage: python eval_nofault.py <task> <ckpt_path> <joint_kp> [n_test]
"""
import sys
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from native_joint_policy import NativeJointPolicy
from favor_fault_runner import FaultRobomimicImageRunner

TASK, CKPT, KP = sys.argv[1], sys.argv[2], int(sys.argv[3])
N_TEST = int(sys.argv[4]) if len(sys.argv) > 4 else 20
DATASET = f"/workspace/data/robomimic/datasets/libero_{TASK}/ph/image_abs.hdf5"

payload = torch.load(open(CKPT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=f"/workspace/results/_evalnf_{TASK}")
ws.load_payload(payload, exclude_keys=None, include_keys=None)
base = ws.ema_model if cfg.training.use_ema else ws.model
base.to(torch.device("cuda:0")); base.eval()

runner = FaultRobomimicImageRunner(
    output_dir=f"/workspace/results/_evalnf_run_{TASK}_{KP}",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=0, n_test=N_TEST, test_start_seed=10000, n_envs=5,
    max_steps=400, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
    render_obs_key="agentview_image", abs_action=True,
    actuation_mode="joint", joint_kp=KP,
)
log = runner.run(NativeJointPolicy(base, base_seed=42, fault_spec=None))
print(f"RESULT task={TASK} kp={KP} ckpt={CKPT.split('/')[-1]} "
      f"n_test={N_TEST} mean_score={log.get('test/mean_score')}")
