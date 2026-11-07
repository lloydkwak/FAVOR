"""Parallel policy rollouts in the pouring env (used by the in-training env runner and by
scripts/m9_eval.py).

Physics runs in `n_workers` env processes (spawn); the policy runs in the calling process, one
episode at a time, with the noise seeded by (episode seed, policy call index) exactly as in the
LIBERO runs and scripts/m8_rollout.py. Episodes are therefore paired across methods and fault
conditions: until the executed actions differ, every method sees the same samples. One iteration
= infer a chunk for every active episode, send all chunks, let all workers simulate in parallel.

Episode spec (dict): seed, grid_index or bowl_xy, fault (None or {joint: 1..7, kind: locked|range,
half_width, onset}), method (favor_xarm.correction.METHODS key).

Early stop (default on): a trial ends once the cup has been tilted past 60 deg, is back below
10 deg and the arm has been still (every joint < 0.005 rad per step) for 1 s; otherwise it runs
to task.max_steps (300 = 30 s). Weighing follows after task.settle_after, as in the scripted demos.
"""
import collections
import math
import multiprocessing as mp
import time

import numpy as np

IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


# ----------------------------------------------------------------------------- env worker
def _worker(conn, cfg, n_obs, early_stop):
    from .env import PourEnv
    from .faults import FaultLayer
    env = PourEnv(cfg, render=True)
    max_steps = int(cfg["task"]["max_steps"])
    hist, still, poured = None, 0, False
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            f = arg.get("fault")
            env.fault = None if not f else FaultLayer(int(f["joint"]) - 1, f.get("kind", "locked"),
                                                      half_width=f.get("half_width", 0.05), onset=f.get("onset"))
            obs = env.reset(seed=int(arg["seed"]), grid_index=arg.get("grid_index"), bowl_xy=arg.get("bowl_xy"))
            hist = collections.deque([obs] * n_obs, maxlen=n_obs)
            still, poured = 0, False
            conn.send((list(hist), env.fault.get_fault_info() if env.fault else None, [float(v) for v in env.bowl_xy]))
        elif cmd == "step":
            done = False
            for q in arg:
                q_prev = env.joint_pos()
                obs = env.step(q)
                hist.append(obs)
                tilt = env.cup_tilt()
                poured |= tilt > math.radians(60)
                moving = np.abs(env.joint_pos() - q_prev).max() > 0.005
                still = 0 if (moving or tilt > math.radians(10)) else still + 1
                if env.step_count >= max_steps or (early_stop and poured and still >= 10):
                    done = True
                    break
            conn.send((list(hist), done, env.fault.get_fault_info() if env.fault else None, env.step_count))
        elif cmd == "finish":
            env.hold(cfg["task"]["settle_after"])
            m = env.metrics()
            if env.fault is not None:
                env.fault.restore(env)
            conn.send(m)
        elif cmd == "close":
            env.close()
            conn.close()
            return


class RolloutPool:
    def __init__(self, cfg, n_workers, n_obs, early_stop=True):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, cfg, n_obs, early_stop), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


# ----------------------------------------------------------------------------- policy side
def set_sampler(policy, sampler="ddpm", steps=None):
    """sampler 'ddpm' keeps the trained scheduler (100 steps unless `steps`); 'ddim' replaces it by a
    DDIM scheduler with the same noise schedule (the real-robot plan: train DDPM 100, infer DDIM 10)."""
    if sampler == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        c = dict(policy.noise_scheduler.config)
        policy.noise_scheduler = DDIMScheduler(
            num_train_timesteps=c["num_train_timesteps"], beta_start=c["beta_start"], beta_end=c["beta_end"],
            beta_schedule=c["beta_schedule"], clip_sample=c.get("clip_sample", True), set_alpha_to_one=True,
            steps_offset=0, prediction_type=c.get("prediction_type", "epsilon"))
        policy.num_inference_steps = steps or 10
    elif steps is not None:
        policy.num_inference_steps = steps
    return policy


def _obs_batch(hist, keys, device):
    import torch
    out = {}
    for k in keys:
        x = np.stack([h[k] for h in hist])
        if k in IMAGE_KEYS:
            x = np.moveaxis(x, -1, 1).astype(np.float32) / 255.0
        out[k] = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(device)
    return out


def run_episodes(policy, pool, episodes, obs_keys, n_action_steps, kin=None, log=print):
    """Runs every episode spec; returns one result dict per episode (same order)."""
    import torch
    from .correction import correct_chunk
    device = next(policy.parameters()).device
    queue = list(enumerate(episodes))
    active = {}                                  # worker index -> state
    results = [None] * len(episodes)
    free = list(range(len(pool.conns)))
    t_start = time.time()

    def start(w):
        i, ep = queue.pop(0)
        pool.conns[w].send(("reset", ep))
        hist, finfo, bowl_xy = pool.conns[w].recv()
        active[w] = dict(i=i, ep=ep, hist=hist, finfo=finfo, call=0, t0=time.time(), t_inf=0.0, n_corr=0,
                         corr=[], bowl_xy=bowl_xy, steps=0)

    while queue or active:
        while free and queue:
            start(free.pop(0))
        sent = []
        for w, st in active.items():
            torch.manual_seed(int(st["ep"]["seed"]) * 100003 + st["call"])
            ti = time.time()
            with torch.no_grad():
                act = policy.predict_action(_obs_batch(st["hist"], obs_keys, device))["action"][0, :n_action_steps]
            st["t_inf"] += time.time() - ti
            st["call"] += 1
            act, lg = correct_chunk(kin, act.detach().cpu(), st["finfo"], st["ep"].get("method", "B1"))
            if lg:
                st["n_corr"] += lg["n"]
                st["corr"].append(lg)
            pool.conns[w].send(("step", act.numpy()))
            sent.append(w)
        for w in sent:
            st = active[w]
            st["hist"], done, st["finfo"], st["steps"] = pool.conns[w].recv()
            if done:
                pool.conns[w].send(("finish", None))
                m = pool.conns[w].recv()
                ep = st["ep"]
                corr = st["corr"]
                results[st["i"]] = dict(
                    episode=st["i"], seed=ep["seed"], grid_index=ep.get("grid_index"),
                    bowl_x=st["bowl_xy"][0], bowl_y=st["bowl_xy"][1], method=ep.get("method", "B1"),
                    fault=None if not ep.get("fault") else
                    f"{ep['fault'].get('kind', 'locked')} J{ep['fault']['joint']}" + (f" {ep['fault']['onset']}" if ep['fault'].get('onset') else ""),
                    steps=st["steps"], calls=st["call"], wall_s=time.time() - st["t0"],
                    inference_s_per_call=st["t_inf"] / max(st["call"], 1), n_corrected_waypoints=st["n_corr"],
                    corr_pos_after_mm=float(np.mean([c["pos_after_mm"] for c in corr])) if corr else None,
                    corr_dq=float(np.mean([c["dq"] for c in corr])) if corr else None,
                    **{k: m[k] for k in ("f", "spill_g", "cup_g", "success", "safety_violation", "max_tilt_deg")},
                    safety_first=str(m["safety_first"]))
                del active[w]
                free.append(w)
                if log:
                    r = results[st["i"]]
                    done_n = sum(x is not None for x in results)
                    log(f"[{done_n}/{len(episodes)}] seed {r['seed']} cell {r['grid_index']} {r['fault'] or 'no fault'} "
                        f"{r['method']}: f={r['f']:.3f} success={int(r['success'])} steps={r['steps']} "
                        f"({time.time() - t_start:.0f}s)")
    return results
