"""
Thin wrapper around the official eval.py — identical logic, with one addition:
an --n_envs CLI flag that overrides cfg.task.env_runner.n_envs before
hydra.utils.instantiate(...). Not needed for Phase 2 (n_envs=28 default fit
GPU with room to spare), kept for Phase 5 in case full-scale sweeps need it.
"""
import sys, os, pathlib, click, hydra, torch, dill, json
sys.path.insert(0, "/workspace/diffusion_policy")
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-n', '--n_envs', default=None, type=int, help='Override cfg.task.env_runner.n_envs')
def main(checkpoint, output_dir, device, n_envs):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True, default=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    if n_envs is not None:
        print(f"[eval_favor] overriding n_envs -> {n_envs} (was {cfg.task.env_runner.get('n_envs')})")
        cfg.task.env_runner.n_envs = n_envs

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    json_log = {k: v for k, v in runner_log.items()}
    out_path = os.path.join(output_dir, 'eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, default=str, sort_keys=True)
    print(f"[eval_favor] wrote {out_path}")

if __name__ == '__main__':
    main()
