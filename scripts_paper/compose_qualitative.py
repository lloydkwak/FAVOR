"""
Compose qualitative figures and the supplementary video from render_qualitative.py output.

Inputs  (results/qual/): scenarios.json, verify.json, snapshots.pkl, <scenario>/<method>/seed<S>*
Outputs:
  paper/figs/fig_setup.{pdf,png}             4 task start scenes + Panda with joints labelled
  paper/figs/fig_qual_<id>.{pdf,png}          methods x keyframes, last column = EE path
  paper/figs/fig_qual_<id>_traj.{pdf,png}     EE paths of all methods: camera overlay + top view
  paper/video/favor_supp.mp4                  title, setup, every scenario side by side (2x2)
  paper/video/<id>_2x2.mp4                    each scenario alone
  results/qual/compose_report.md              what was used, reproduction check per episode

Keyframes use the same absolute steps for every method, taken from the Priority IK
episode: grasp (first gripper-close command), midway, success (first reward 1).

Needs numpy, matplotlib, pillow, av (all in the LIBERO container).
Usage: python scripts_paper/compose_qualitative.py [--qual results/qual] [--out paper] [--seed-index 0]
"""
import argparse, json, os, pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

TASK_LABEL = {"alphabet_soup": "Soup", "milk": "Milk", "bowl_ramekin": "Bowl-Ramekin", "bowl_stove": "Bowl-Stove"}
TASK_DESC = {"alphabet_soup": "pick the alphabet soup and place it in the basket",
             "milk": "pick the milk and place it in the basket",
             "bowl_ramekin": "pick the black bowl between the plate and the ramekin, place it on the plate",
             "bowl_stove": "pick the black bowl on the stove and place it on the plate"}
LABEL = {"b1": "B1 (no intervention)", "pos": "B-IK pos", "pose": "B-IK pose", "prio": "Priority IK (ours)",
         "rg": "RG-DDPM"}
COLOR = {"b1": "#9E9C9C", "rg": "#5B7DB8", "pos": "#E39A9E", "pose": "#F2C4C6", "prio": "#B4131C"}
LINE = {"b1": "#8A8A8A", "rg": "#3E64A8", "pos": "#E07A80", "pose": "#C98A8E", "prio": "#B4131C"}  # paths on images
LS = {"pose": (0, (3, 1.5))}
OK, BAD = "#1E8C3A", "#B4131C"
FPS = 20

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})


# ---------------------------------------------------------------- data
def project(points, cam):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pc = (p - cam["cam_pos"]) @ cam["cam_mat"]
    z = np.maximum(-pc[:, 2], 1e-6)
    f = 0.5 * cam["height"] / np.tan(np.deg2rad(cam["fovy"]) / 2)
    return np.stack([cam["width"] / 2 + f * pc[:, 0] / z, cam["height"] / 2 - f * pc[:, 1] / z], 1)


class Ep:
    def __init__(self, d, seed):
        b = os.path.join(d, f"seed{seed}")
        t = np.load(b + "_trace.npz", allow_pickle=True)
        self.ee, self.act, self.rew = t["ee"], t["act"], t["rew"]
        self.success = float(t["success"])
        self.cam = {k: t[k] for k in ("cam_pos", "cam_mat")}
        self.cam.update(fovy=float(t["fovy"]), width=int(t["width"]), height=int(t["height"]))
        fr = np.load(b + "_frames.npz")
        self.frames, self.steps = fr["frames"], list(fr["steps"])
        self.mp4 = b + ".mp4"
        self.uv = project(self.ee, self.cam)

    def frame(self, step):
        i = int(np.argmin([abs(s - step) for s in self.steps]))
        return self.frames[i]

    def t_success(self):
        hit = np.nonzero(self.rew >= 1)[0]
        return int(hit[0]) + 1 if len(hit) else None

    def t_grasp(self):
        hit = np.nonzero(self.act[:, -1] > 0)[0]
        return int(hit[0]) + 1 if len(hit) else None


def keyframes(ref, stride):
    T = len(ref.rew)
    ts = ref.t_success() or T
    tg = ref.t_grasp()
    if tg is None or tg >= ts:
        tg = ts // 3
    snap = lambda s: int(min(T, max(0, round(s / stride) * stride)))
    return [snap(tg), snap((tg + ts) / 2), snap(ts)]


def crop_box(eps, W, H, aspect=4 / 3, pad=0.35, min_frac=0.55):
    uv = np.concatenate([e.uv for e in eps])
    uv = uv[(uv[:, 0] > -W) & (uv[:, 0] < 2 * W) & (uv[:, 1] > -H) & (uv[:, 1] < 2 * H)]
    lo, hi = uv.min(0), uv.max(0)
    c = (lo + hi) / 2
    w = max((hi[0] - lo[0]) * (1 + 2 * pad), (hi[1] - lo[1]) * (1 + 2 * pad) * aspect, min_frac * W)
    w = min(w, W); h = min(w / aspect, H); w = h * aspect
    x0 = float(np.clip(c[0] - w / 2, 0, W - w)); y0 = float(np.clip(c[1] - h / 2, 0, H - h))
    return int(x0), int(y0), int(x0 + w), int(y0 + h)


# ---------------------------------------------------------------- paper figures
def fig_qual(sc, eps, methods, out, stride):
    ref = eps["prio"]
    ks = keyframes(ref, stride)
    W, H = ref.cam["width"], ref.cam["height"]
    x0, y0, x1, y1 = crop_box([eps[m] for m in methods], W, H)
    ncol = len(ks) + 1
    pw = 1.55
    ph = pw * (y1 - y0) / (x1 - x0)
    fig, axs = plt.subplots(len(methods), ncol, figsize=(0.55 + pw * ncol, 0.25 + ph * len(methods)),
                            gridspec_kw=dict(wspace=0.03, hspace=0.05), squeeze=False)
    for r, m in enumerate(methods):
        e = eps[m]
        for c in range(ncol):
            ax = axs[r, c]
            if c < len(ks):
                ax.imshow(e.frame(ks[c])[y0:y1, x0:x1])
                if r == 0:
                    lab = ["grasp", "transport", "place"][c] if c < 3 else ""
                    ax.set_title(f"t = {ks[c] / FPS:.1f} s  ({lab})", pad=2)
            else:
                ax.imshow(e.frame(e.steps[-1])[y0:y1, x0:x1])
                uv = e.uv - [x0, y0]
                ax.plot(uv[:, 0], uv[:, 1], color="white", lw=2.6, alpha=0.9)
                ax.plot(uv[:, 0], uv[:, 1], color=LINE[m], lw=1.4)
                ax.plot(*uv[0], "o", ms=3.5, mfc="white", mec="black", mew=0.6)
                ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
                if r == 0:
                    ax.set_title(f"end-effector path (t = {e.steps[-1] / FPS:.0f} s)", pad=2)
                ok = e.success >= 1
                ax.text(0.97, 0.05, "success" if ok else "fail", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc=OK if ok else BAD, ec="none"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(1.6 if m == "prio" else 0.4)
                s.set_color(COLOR["prio"] if m == "prio" else "#999999")
        axs[r, 0].set_ylabel(LABEL[m].replace(" (", "\n("), fontsize=8,
                             fontweight="bold" if m == "prio" else "normal")
    fig.suptitle(f"{TASK_LABEL[sc['task']]}, joint {sc['joint']} locked (seed {sc['seed']})", y=1.0, fontsize=9)
    save(fig, out)


def fig_traj(sc, eps, methods, out):
    ref = eps["prio"]
    W, H = ref.cam["width"], ref.cam["height"]
    x0, y0, x1, y1 = crop_box([eps[m] for m in methods], W, H)
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw=dict(width_ratios=[(x1 - x0) / (y1 - y0), 1.1]))
    a.imshow(ref.frame(0)[y0:y1, x0:x1])
    for m in methods:
        uv = eps[m].uv - [x0, y0]
        a.plot(uv[:, 0], uv[:, 1], color="white", lw=2.4, alpha=0.8)
        a.plot(uv[:, 0], uv[:, 1], color=LINE[m], lw=1.2, ls=LS.get(m, "-"), label=LABEL[m])
    a.plot(*(ref.uv[0] - [x0, y0]), "o", ms=3.5, mfc="white", mec="black", mew=0.6)
    a.set_xlim(0, x1 - x0); a.set_ylim(y1 - y0, 0); a.set_xticks([]); a.set_yticks([])
    for m in methods:
        e = eps[m].ee
        b.plot(e[:, 0], e[:, 1], color=LINE[m], lw=1.6 if m == "prio" else 1.0, ls=LS.get(m, "-"), label=LABEL[m])
        b.plot(e[-1, 0], e[-1, 1], "x" if eps[m].success < 1 else "o", color=LINE[m], ms=4)
    b.plot(ref.ee[0, 0], ref.ee[0, 1], "o", ms=3.5, mfc="white", mec="black", mew=0.6)
    b.set_aspect("equal", "datalim"); b.set_xlabel("x [m]"); b.set_ylabel("y [m]")
    b.tick_params(labelsize=6); b.spines["top"].set_visible(False); b.spines["right"].set_visible(False)
    b.set_title("top view", pad=2)
    h, l = b.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    save(fig, out)


def fig_setup(snaps, out, joint):
    tasks = [t for t in TASK_LABEL if t in snaps]
    fig, axs = plt.subplots(1, len(tasks) + 1, figsize=(7.16, 1.65), gridspec_kw=dict(wspace=0.04))
    for ax, t in zip(axs, tasks):
        ax.imshow(snaps[t]["plain"]["img"]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(TASK_LABEL[t], pad=2)
    t0 = tasks[-1]
    s = snaps[t0]["plain"]
    img = snaps[t0]["highlight"][joint]
    cam = {k: s[k] for k in ("cam_pos", "cam_mat", "fovy", "width", "height")}
    uv = project(s["anchors"], cam)
    ee = project(s["ee"][None], cam)[0]
    pts = np.vstack([uv, ee])
    W, H = cam["width"], cam["height"]
    lo, hi = pts.min(0), pts.max(0)
    side = max(hi - lo) * 1.9
    c = (lo + hi) / 2 + [side * 0.12, 0]
    x0 = int(np.clip(c[0] - side / 2, 0, W - min(side, W))); y0 = int(np.clip(c[1] - side / 2, 0, H - min(side, H)))
    x1, y1 = int(min(W, x0 + side)), int(min(H, y0 + side))
    ax = axs[-1]
    ax.imshow(img[y0:y1, x0:x1]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Panda, J{joint} in red", pad=2)
    q = uv - [x0, y0]
    lx = (x1 - x0) * 0.97
    ys = np.linspace((y1 - y0) * 0.1, (y1 - y0) * 0.9, 7)
    order = np.argsort(q[:, 1])
    for k, i in enumerate(order):
        col = COLOR["prio"] if i + 1 == joint else "black"
        ax.plot(*q[i], "o", ms=2.5, color=col)
        ax.annotate(f"J{i + 1}", xy=q[i], xytext=(lx, ys[k]), fontsize=6.5, ha="right", va="center", color=col,
                    fontweight="bold" if i + 1 == joint else "normal",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=col, shrinkA=1, shrinkB=1))
    ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
    for a in axs:
        for sp in a.spines.values():
            sp.set_linewidth(0.4); sp.set_color("#999999")
    save(fig, out)


def save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote", out + ".pdf/.png")


# ---------------------------------------------------------------- video
def font(size, bold=False):
    p = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


class Writer:
    def __init__(self, path, W, H, crf=18):
        import av
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.av, self.c = av, av.open(path, mode="w")
        self.s = self.c.add_stream("h264", rate=FPS)
        self.s.width, self.s.height, self.s.pix_fmt = W, H, "yuv420p"
        self.s.options = {"crf": str(crf)}
        self.W, self.H, self.path = W, H, path

    def put(self, img, n=1):
        f = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(img[:self.H, :self.W]), format="rgb24")
        for _ in range(n):
            for p in self.s.encode(f):
                self.c.mux(p)

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
        print("wrote", self.path)


def card(W, H, lines):
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    y = H * 0.36
    for text, size, bold, col in lines:
        f = font(size, bold)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=col)
        y += size * 1.6
    return np.array(im)


def decode(path):
    import av
    with av.open(path) as c:
        for fr in c.decode(video=0):
            yield fr.to_ndarray(format="rgb24")


def tile_frames(sc, eps, methods, tile=480, head=72):
    """Yield 2x2 (or 1xN) composited frames for one scenario, with labels, EE trail and outcome."""
    cols = 2 if len(methods) == 4 else len(methods)
    rows = (len(methods) + cols - 1) // cols
    W, H = cols * tile, rows * tile + head
    gens = {m: decode(eps[m].mp4) for m in methods}
    last = {m: None for m in methods}
    T = max(len(eps[m].rew) for m in methods) + 1
    fb, fs, fh = font(22, True), font(18), font(24, True)
    for t in range(T + FPS * 2):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        hdr = f"{TASK_LABEL[sc['task']]}, joint {sc['joint']} locked (red link), seed {sc['seed']}"
        d.text((16, 10), hdr, font=fh, fill=(20, 20, 20))
        d.text((16, 42), TASK_DESC[sc["task"]], font=fs, fill=(90, 90, 90))
        tt = f"t = {min(t, T - 1) / FPS:4.1f} s"
        d.text((W - 16 - d.textlength(tt, font=fh), 10), tt, font=fh, fill=(20, 20, 20))
        for k, m in enumerate(methods):
            e = eps[m]
            if t < T:
                try:
                    last[m] = next(gens[m])
                except StopIteration:
                    pass
            fr = Image.fromarray(last[m]).resize((tile, tile))
            ox, oy = (k % cols) * tile, head + (k // cols) * tile
            canvas.paste(fr, (ox, oy))
            s = tile / e.cam["width"]
            n = min(t, len(e.uv) - 1)
            pts = [(ox + u * s, oy + v * s) for u, v in e.uv[:n + 1]]
            if len(pts) > 1:
                d.line(pts, fill=(255, 255, 255), width=6)
                d.line(pts, fill=hex2rgb(LINE[m]), width=3)
            lab = LABEL[m]
            w = d.textlength(lab, font=fb)
            d.rectangle([ox + 8, oy + 8, ox + 22 + w, oy + 40], fill=(255, 255, 255))
            d.rectangle([ox + 8, oy + 8, ox + 13, oy + 40], fill=hex2rgb(COLOR[m]))
            d.text((ox + 18, oy + 11), lab, font=fb, fill=(20, 20, 20))
            ts = e.t_success()
            if ts is not None and t >= ts:
                badge, col = "SUCCESS", OK
            elif t >= len(e.rew):
                badge, col = "FAIL", BAD
            else:
                badge = None
            if badge:
                w = d.textlength(badge, font=fb)
                d.rectangle([ox + tile - 22 - w, oy + tile - 44, ox + tile - 8, oy + tile - 10], fill=hex2rgb(col))
                d.text((ox + tile - 15 - w, oy + tile - 41), badge, font=fb, fill=(255, 255, 255))
            d.rectangle([ox, oy, ox + tile - 1, oy + tile - 1], outline=(255, 255, 255), width=2)
        yield np.array(canvas)


def setup_frames(snaps, W, H):
    tasks = [t for t in TASK_LABEL if t in snaps]
    tile = min(W // 2, (H - 72) // 2)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((16, 10), "Four LIBERO tasks, Franka Panda", font=font(24, True), fill=(20, 20, 20))
    d.text((16, 42), "one pretrained joint-space diffusion policy per task", font=font(18), fill=(90, 90, 90))
    for k, t in enumerate(tasks):
        ox, oy = (k % 2) * tile + (W - 2 * tile) // 2, 72 + (k // 2) * tile
        canvas.paste(Image.fromarray(snaps[t]["plain"]["img"]).resize((tile, tile)), (ox, oy))
        lab = TASK_LABEL[t]
        f = font(22, True)
        w = d.textlength(lab, font=f)
        d.rectangle([ox + 8, oy + 8, ox + 20 + w, oy + 40], fill=(255, 255, 255))
        d.text((ox + 14, oy + 11), lab, font=f, fill=(20, 20, 20))
    return np.array(canvas)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qual", default="results/qual")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--seed-index", type=int, default=0, help="which selected seed per scenario")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--setup-joint", type=int, default=0, help="highlighted joint in fig_setup (0: scenario A's)")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    Q = a.qual
    scen = json.load(open(os.path.join(Q, "scenarios.json")))
    verify = json.load(open(os.path.join(Q, "verify.json"))) if os.path.exists(os.path.join(Q, "verify.json")) else {}
    report = ["# Qualitative media report", ""]
    loaded = []
    for sc in scen:
        sc = dict(sc)
        seed = sc["seeds"][min(a.seed_index, len(sc["seeds"]) - 1)]
        sc["seed"] = seed
        eps, rows = {}, []
        for m in ["b1", "pos", "pose", "rg", "prio"]:
            d = os.path.join(Q, sc["id"], m)
            if not os.path.exists(os.path.join(d, f"seed{seed}_frames.npz")):
                continue
            eps[m] = Ep(d, seed)
            v = verify.get(f"{sc['id']}:{m}", {})
            o = (v.get("orig") or {}).get(str(seed))
            match = "n/a" if o is None else ("yes" if float(o) == eps[m].success else "NO")
            rows.append(f"| {LABEL[m]} | {o} | {eps[m].success:.0f} | {match} | "
                        f"{'yes' if v.get('match_all') else 'no' if v else 'n/a'} |")
        report += [f"## {sc['id']}: {TASK_LABEL[sc['task']]} J{sc['joint']} locked, seed {seed}", "",
                   "| method | sweep JSON | re-run | match | all re-run episodes match |",
                   "|---|---|---|---|---|"] + rows + [""]
        if "prio" not in eps:
            print(f"[{sc['id']}] no Priority IK render, skipped"); continue
        loaded.append((sc, eps))

    figs = os.path.join(a.out, "figs")
    for sc, eps in loaded:
        if sc["id"].startswith("C"):
            continue
        shown = [m for m in ["b1"] + sc["contrast"] + ["prio"] if m in eps]
        fig_qual(sc, eps, shown, os.path.join(figs, f"fig_qual_{sc['id']}"), a.stride)
        fig_traj(sc, eps, [m for m in ["b1", "pos", "pose", "prio"] if m in eps],
                 os.path.join(figs, f"fig_qual_{sc['id']}_traj"))

    snaps = None
    sp = os.path.join(Q, "snapshots.pkl")
    if os.path.exists(sp):
        snaps = pickle.load(open(sp, "rb"))
        j = a.setup_joint or (loaded[0][0]["joint"] if loaded else 7)
        fig_setup(snaps, os.path.join(figs, "fig_setup"), j)

    if not a.no_video and loaded:
        vids = os.path.join(a.out, "video")
        methods = lambda eps: [m for m in ["b1", "pos", "pose", "prio"] if m in eps]
        W, H = 960, 1032
        sup = Writer(os.path.join(vids, "favor_supp.mp4"), W, H)
        sup.put(card(W, H, [("Training-free fault adaptation", 40, True, (20, 20, 20)),
                            ("of a joint-space diffusion policy", 40, True, (20, 20, 20)),
                            ("", 20, False, (0, 0, 0)),
                            ("one joint locked at its initial angle, no retraining", 26, False, (90, 90, 90)),
                            ("same seed and same policy samples for every method", 26, False, (90, 90, 90))]), FPS * 4)
        if snaps:
            sup.put(setup_frames(snaps, W, H), FPS * 4)
        sup.put(card(W, H, [("Methods", 34, True, (20, 20, 20)), ("", 16, False, (0, 0, 0)),
                            ("B1: policy output sent as is", 26, False, (90, 90, 90)),
                            ("B-IK pos / pose: weighted IK on the healthy joints", 26, False, (90, 90, 90)),
                            ("Priority IK: position first, orientation in its null space", 26, True, hex2rgb(COLOR["prio"]))]),
                FPS * 5)
        for sc, eps in loaded:
            ms = methods(eps)
            one = Writer(os.path.join(vids, f"{sc['id']}_2x2.mp4"), W, H)
            sup.put(card(W, H, [(f"{TASK_LABEL[sc['task']]}, joint {sc['joint']} locked", 38, True, (20, 20, 20)),
                                (TASK_DESC[sc["task"]], 26, False, (90, 90, 90))]), int(FPS * 2.5))
            for fr in tile_frames(sc, eps, ms):
                sup.put(fr); one.put(fr)
            one.close()
        sup.close()

    with open(os.path.join(Q, "compose_report.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
