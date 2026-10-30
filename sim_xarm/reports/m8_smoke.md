# M8: pipeline smoke test (no performance target)

Data: `out/xarm_pour_200.hdf5` (M5, 200 scripted demonstrations). Machine: 2 CPU cores, 8 GB RAM,
no GPU — so the policy is **deliberately small** and trained for only 300 steps; this checks that
every stage connects, nothing else.

## Training (upstream diffusion_policy `train_diffusion_unet_hybrid_workspace`, task `xarm_pour_image_joint`)

- Same structure as the LIBERO runs: CNN U-Net, ε-prediction, DDPM 100 steps, horizon 16, n_obs 2,
  n_action 8, ResNet-18 image encoders (crop 76), EMA.
- Changed for this machine only: `policy.down_dims=[64,128,256]` (the default [512,1024,2048], 255 M
  parameters, was killed for memory), batch 32, 2 epochs × 150 steps, `lr_warmup_steps=50`,
  checkpoint on train_loss, `NullRunner` (no in-training rollouts).
- Loss 1.14 (step 1) → 0.74 (step 100) → 0.33 (step 250); epoch means 0.817 / 0.338; 3.5–3.9 s per step.
  Checkpoints written (`epoch=0001-train_loss=0.3379.ckpt`, `latest.ckpt`).

Command used (from `diffusion_policy/`, `PYTHONPATH=sim_xarm/dp:sim_xarm:.`):

```
python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml task=xarm_pour_image_joint \
  task.dataset_path=.../xarm_pour_200.hdf5 training.device=cpu logging.mode=disabled \
  dataloader.num_workers=0 val_dataloader.num_workers=0 dataloader.batch_size=32 val_dataloader.batch_size=32 \
  "policy.down_dims=[64,128,256]" training.num_epochs=2 training.max_train_steps=150 training.max_val_steps=5 \
  training.lr_warmup_steps=50 training.checkpoint_every=1 training.val_every=1 training.sample_every=100 \
  training.rollout_every=100 checkpoint.topk.monitor_key=train_loss checkpoint.topk.mode=min \
  "checkpoint.topk.format_str='epoch={epoch:04d}-train_loss={train_loss:.4f}.ckpt'"
```

## Rollouts (`scripts/m8_rollout.py`, 300 steps = 30 s, 38 policy calls each, DDPM 100 steps)

| condition | method | cell | steps | waypoints corrected | correction residual (mm) | healthy-joint dq (rad) | f | safety (first) |
|---|---|---|---|---|---|---|---|---|
| no fault | – | 12 | 300 | 0 | – | – | 0.00 | self collision at 2.1 s |
| no fault | – | 0 | 300 | 0 | – | – | 0.03 | table contact at 4.0 s |
| locked J6 | B1 | 12 | 300 | 0 | – | – | 0.00 | table contact at 1.1 s |
| locked J6 | Priority IK | 12 | 300 | 304 | 14.7 | 0.97 | 0.00 | table contact at 6.7 s |

- Every stage runs end to end: checkpoint load → image + eef observations from the env → policy
  chunk → fault layer / correction → 10 Hz joint targets → physics → weighing. Inference 1.03–1.07 s
  per call on CPU (small U-Net).
- As expected for 300 training steps of a small model, the policy does not pour (it tilts the cup to
  115–137° but not over the bowl and hits the table). B1 corrects nothing; Priority IK corrects all
  304 executed waypoints with the paper's parameters (λ₁ 0.01, λ₂ 0.2, 30 iterations).
- Not done here (next session): full-size training on the GPU server, the no-fault gate
  (20 rollouts, success ≥ 0.75), the fault sweep. Raw rows: `reports/m8_rollouts.jsonl`.
