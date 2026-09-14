"""
Evaluate ONE checkpoint with ONE FaultRobomimicImageRunner construction, in
its own process. Deliberately not looped over multiple checkpoints/kp values
in a single Python process: every attempt to construct a second
FaultRobomimicImageRunner (or call env_fns[0]() a second time) in the same
process has deadlocked (confirmed three times: 0% CPU, no progress, process
alive) after the first one succeeds. One process per checkpoint, invoked
from the shell in a loop, is the pattern that has actually worked so far.

Usage: python eval_checkpoint.py <ckpt_path> <n_test> <joint_kp>
"""
import sys
import torch
import dill
import hydra

from favor_fault_runner import FaultRobomimicImageRunner

ckpt_path, n_test, joint_kp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
cfg = payload["cfg"]
cls = hydra.utils.get_class(cfg._target_)
ws = cls(cfg, output_dir=None)
ws.load_payload(payload, exclude_keys=None, include_keys=None)
policy = ws.ema_model if cfg.training.use_ema else ws.model
policy.eval().to("cuda:0")

runner = FaultRobomimicImageRunner(
    output_dir="/tmp/eval_ckpt",
    dataset_path=cfg.task.env_runner.dataset_path if "env_runner" in cfg.task else cfg.task.dataset.dataset_path,
    shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint1", fault_type=None, fault_severity=None,
    n_train=0, n_test=n_test, n_envs=min(5, n_test), test_start_seed=10000, max_steps=400,
    n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
    render_obs_key="agentview_image", abs_action=True, actuation_mode="joint",
    joint_kp=joint_kp,
)
log = runner.run(policy)
print("CKPT=%s KP=%d N=%d SCORE=%.4f" % (ckpt_path, joint_kp, n_test, log.get("test/mean_score", -1)))
