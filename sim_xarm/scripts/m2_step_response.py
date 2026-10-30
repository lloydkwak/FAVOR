"""M2: per-joint step response of the simulated position servo through the command layer.

For each joint (others held at home) a +step_deg target is issued at t = 0.5 s:
  actuator : the servo ctrl is stepped directly (plant only, no command shaping)
  servo    : Mode 1 emulation (10 Hz target -> 100 Hz linear stream, 180 deg/s clamp)
  online   : Mode 6 emulation (online trapezoid, planner speed / acceleration)
Measured joint angle is logged at the physics rate. Reports rise time (10-90 %), overshoot,
2 % settling time and steady-state error; plot in reports/m2_step_response.png.
Particles are omitted (their 50 g payload changes the arm dynamics negligibly)."""
import argparse
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
import numpy as np

from favor_xarm.config import add_cli, from_args, resolve
from favor_xarm.control import JointCommandStreamer
from favor_xarm.scene import build_scene

T0, T_END = 0.5, 2.0


def run(cfg, h, j, step, mode):
    m = h.model
    d = mujoco.MjData(m)
    q0 = np.array(cfg["robot"]["home_q"], float)
    d.qpos[h.arm_qpos] = q0
    d.ctrl[h.arm_act] = q0
    if h.gripper_act >= 0:
        d.ctrl[h.gripper_act] = cfg["robot"]["gripper"]["closed_ctrl"]
    mujoco.mj_forward(m, d)
    c = copy.deepcopy(cfg)
    if mode in ("servo", "online"):
        c["control"]["mode"] = mode
    ctl = JointCommandStreamer(c, q0)
    tgt = q0.copy(); tgt[j] += step
    dt = m.opt.timestep
    n_sub = int(round(1.0 / cfg["control"]["stream_hz"] / dt))
    per_policy = ctl.ticks_per_step
    t, ts, qs, cmd = 0.0, [], [], []
    tick = 0
    while t < T_END - 1e-9:
        if mode == "actuator":
            u = tgt if t >= T0 else q0
        else:
            if tick % per_policy == 0:
                ctl.set_target(tgt if t >= T0 - 1e-9 else q0)
            u = ctl.tick()
        d.ctrl[h.arm_act] = u
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
            t += dt
            ts.append(t); qs.append(d.qpos[h.arm_qpos[j]]); cmd.append(u[j] if mode == "actuator" else ctl.q_cmd[j])
        tick += 1
    return np.array(ts), np.array(qs), np.array(cmd), q0[j], tgt[j]


def ramp_lag(cfg, h, j, speed, ff):
    """Track a constant-speed ramp of joint j sent as 10 Hz servo targets; mean |q_cmd - q| over
    the last half of a 1 s ramp (deg)."""
    m = h.model
    d = mujoco.MjData(m)
    q0 = np.array(cfg["robot"]["home_q"], float)
    d.qpos[h.arm_qpos] = q0; d.ctrl[h.arm_act] = q0
    mujoco.mj_forward(m, d)
    c = copy.deepcopy(cfg); c["control"]["mode"] = "servo"; c["control"]["velocity_feedforward"] = ff
    ctl = JointCommandStreamer(c, q0)
    n_sub = int(round(1.0 / cfg["control"]["stream_hz"] / m.opt.timestep))
    lags = []
    for k in range(10):
        tgt = q0.copy(); tgt[j] += speed * 0.1 * (k + 1)
        ctl.set_target(tgt)
        for _ in range(ctl.ticks_per_step):
            d.ctrl[h.arm_act] = ctl.tick()
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            if k >= 5:
                lags.append(abs(ctl.q_cmd[j] - d.qpos[h.arm_qpos[j]]))
    return math.degrees(float(np.mean(lags)))


def metrics(t, q, q0, q1):
    y = (q - q0) / (q1 - q0)
    after = t >= T0
    ta, ya = t[after], y[after]
    def first(level):
        idx = np.nonzero(ya >= level)[0]
        return ta[idx[0]] if len(idx) else float("nan")
    rise = first(0.9) - first(0.1)
    over = max(0.0, (ya.max() - 1.0) * 100)
    outside = np.nonzero(np.abs(ya - 1.0) > 0.02)[0]
    settle = (ta[outside[-1]] - T0) if len(outside) else 0.0
    sse = float(np.degrees(np.mean(q[t > T_END - 0.2]) - q1))
    return dict(rise_ms=rise * 1000, t90_ms=(first(0.9) - T0) * 1000, overshoot_pct=over, settle_2pct_ms=settle * 1000,
                sse_deg=sse)


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--step-deg", type=float, default=10.0)
    args = ap.parse_args()
    cfg = from_args(args)
    h = build_scene(cfg, n_particles=0)
    step = math.radians(args.step_deg)
    modes = ["actuator", "servo", "online"]
    res, traces = {}, {}
    for j in range(7):
        for mode in modes:
            t, q, u, q0, q1 = run(cfg, h, j, step, mode)
            res[f"J{j + 1}/{mode}"] = metrics(t, q, q0, q1)
            traces[(j, mode)] = (t, q, u, q0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"actuator": "#2a78d6", "servo": "#eb6834", "online": "#1baf7a"}
    labels = {"actuator": "servo ctrl stepped (plant)", "servo": "Mode 1 emulation (servo)",
              "online": "Mode 6 emulation (online)"}
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for j in range(7):
        ax = axes.flat[j]
        for mode in modes:
            t, q, u, q0 = traces[(j, mode)]
            ax.plot((t - T0) * 1000, np.degrees(q - q0), color=colors[mode], lw=2, label=labels[mode])
        t, q, u, q0 = traces[(j, "servo")]
        ax.plot((t - T0) * 1000, np.degrees(u - q0), color="#8a8a85", lw=1.2, ls="--", label="Mode 1 command stream")
        ax.axhline(args.step_deg, color="#c3c2b7", lw=0.8, zorder=0)
        ax.set_title(f"J{j + 1}", fontsize=11, loc="left")
        ax.set_xlim(-50, 600)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if j % 4 == 0:
            ax.set_ylabel("joint angle change (deg)")
        if j >= 3:
            ax.set_xlabel("time after step (ms)")
    axes.flat[3].tick_params(labelbottom=True)
    axl = axes.flat[7]
    axl.axis("off")
    hdl, lab = axes.flat[0].get_legend_handles_labels()
    axl.legend(hdl, lab, loc="center left", frameon=False, fontsize=10)
    fig.suptitle(f"Simulated xArm7 joint step response, +{args.step_deg:g} deg from home "
                 f"(kp/kv from config, {cfg['robot']['max_joint_speed_deg']:g} deg/s limit)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    out = resolve("reports/m2_step_response.png")
    fig.savefig(out, dpi=130)
    json.dump(res, open(resolve("reports/m2_step_response.json"), "w"), indent=1)
    L = ["| joint / mode | rise 10-90 % (ms) | t90 (ms) | overshoot (%) | settle 2 % (ms) | steady-state err (deg) |",
         "|---|---|---|---|---|---|"]
    for k, v in res.items():
        L.append(f"| {k} | {v['rise_ms']:.1f} | {v['t90_ms']:.1f} | {v['overshoot_pct']:.2f} | {v['settle_2pct_ms']:.0f} | {v['sse_deg']:+.4f} |")
    L += ["", "Tracking lag on a 0.5 rad/s ramp (servo mode, mean |q_cmd - q| over the last 0.5 s, deg):", "",
          "| joint | without feedforward | with velocity feedforward |", "|---|---|---|"]
    for j in range(7):
        a, b = ramp_lag(cfg, h, j, 0.5, False), ramp_lag(cfg, h, j, 0.5, True)
        res[f"J{j + 1}/ramp_lag_deg"] = {"no_ff": a, "ff": b}
        L.append(f"| J{j + 1} | {a:.3f} | {b:.3f} |")
    json.dump(res, open(resolve("reports/m2_step_response.json"), "w"), indent=1)
    open(resolve("reports/m2_step_response.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
