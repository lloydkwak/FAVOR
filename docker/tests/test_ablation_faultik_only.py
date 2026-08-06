"""
Ablation: isolates the contribution of EmbodimentGuidance from the
fault-aware terminal IK alone. Three-way comparison (same seeds):
  (a) B1: unconstrained IK (fault_spec=None)         -- already have: 0.0
  (b) Fault-aware IK only, NO guidance                -- NEW, this script
  (c) Fault-aware IK + EmbodimentGuidance (FAVOR)      -- already have: 0.9
If (b) is already close to 0.9, guidance's marginal contribution is small
(the fault-aware terminal IK alone explains most of the gain). If (b) is
much closer to 0.0, guidance is doing the heavy lifting.
"""
import sys, json
sys.path.insert(0, "/workspace/diffusion_policy")
sys.path.insert(0, "/workspace/docker")
import torch, dill, hydra
from favor_policy import FavorHybridImagePolicy
from favor_fault_runner import FaultRobomimicImageRunner

CKPT = "/workspace/data/checkpoints/lift_ph/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
DATASET = "/workspace/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
TEST_START_SEED = 10000
N_TEST = 20
q_lo = torch.tensor([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973], dtype=torch.float32)
q_hi = torch.tensor([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32)

payload = torch.load(open(CKPT, 'rb'), pickle_module=dill)
cfg = payload['cfg']
workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg, output_dir="/workspace/results/_scratch_ablation")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
base_policy = workspace.ema_model if cfg.training.use_ema else workspace.model
base_policy.to(torch.device("cuda:0"))
base_policy.eval()

runner = FaultRobomimicImageRunner(
    output_dir="/workspace/results/_ablation_faultik_out",
    dataset_path=DATASET, shape_meta=cfg.task.shape_meta,
    fault_joint_name="robot0_joint3", fault_type="locked", fault_severity=None,
    n_train=0, n_test=N_TEST, test_start_seed=TEST_START_SEED, n_envs=N_TEST,
    max_steps=cfg.task.env_runner.max_steps,
    n_obs_steps=cfg.task.env_runner.n_obs_steps,
    n_action_steps=cfg.task.env_runner.n_action_steps,
    render_obs_key=cfg.task.env_runner.render_obs_key,
    abs_action=cfg.task.env_runner.abs_action,
    actuation_mode='joint',
)
# Fault-aware terminal IK (locked_mask honored via fault_spec), NO guidance
fault_spec = {'joint_idx': 2, 'q_lock': 0.0, 'fault_type': 'locked', 'q_lo': q_lo, 'q_hi': q_hi}
favor_noguidance = FavorHybridImagePolicy(base_policy, fault_spec=fault_spec, env_ref=runner.env,
                                           guidance=None,  # <-- the only difference vs FAVOR
                                           actuation_mode='joint', joint_q_lo=q_lo, joint_q_hi=q_hi)
log = runner.run(favor_noguidance)
score = log.get("test/mean_score")
print(f"Fault-aware IK ONLY (no guidance), joint3 locked, n={N_TEST}: {score}")
print("(reference: B1 unconstrained=0.0, FAVOR with guidance=0.9)")

with open("/workspace/results/ablation_faultik_only_summary.json", "w") as f:
    json.dump({"fault_aware_ik_only": score, "n": N_TEST, "seed": TEST_START_SEED}, f, indent=2)
