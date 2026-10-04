"""
RA-L multimedia package from paper/video/favor_supp.mp4:
  paper/multimedia/FAVOR_multimedia.zip   video + ReadMe.txt + Summary.txt  (RA-L limit: one zip, 50 MB)
  paper/multimedia/favor_supp_10MB.mp4    re-encoded to fit a 10 MB single-video upload (ICRA option)
Needs av (LIBERO container). Usage: python scripts_paper/package_multimedia.py [--video paper/video] [--out paper/multimedia]
"""
import argparse, os, zipfile

README = """FAVOR supplementary video
==========================

File: favor_supp.mp4 (H.264, {w}x{h}, 20 fps, {dur:.0f} s, no audio)
Plays in any standard video player (VLC, QuickTime, Windows Media Player, web browsers).

Contents
- Title and the four LIBERO tasks with the Franka Panda.
- Locked-joint scenarios, each shown side by side for four methods (B1, W-IK pos, W-IK pose,
  Priority IK) in real time. Every panel uses the same seed and the same diffusion samples;
  only the execution-time correction differs. The link attached to the locked joint is red,
  the end-effector path is overlaid, and a badge marks success or failure.
{scen}
All shown episodes are re-rendered from the evaluation sweep and reproduce its recorded
outcomes. Code: https://github.com/lloydkwak/FAVOR
"""

SUMMARY = """The video shows a pretrained joint-space diffusion policy on a Franka Panda with one joint locked,
corrected at inference time without retraining. For each scenario, the same episode is executed
with no correction (B1), weighted inverse kinematics with two orientation weights (W-IK pos, W-IK
pose), and position-first task-priority IK (Priority IK). It illustrates the paper's finding that
how the residual end-effector error is split between position and orientation decides the outcome,
and closes with a fault that no inference-time correction can recover.
"""


def reencode(src, dst, target_mb):
    """Single streaming pass (frames are never all held in memory)."""
    import av
    ic = av.open(src)
    s = ic.streams.video[0]
    dur = float(ic.duration / 1e6)
    bitrate = int(target_mb * 8e6 * 0.92 / dur)           # 8 % container/rate-control margin
    o = av.open(dst, mode="w")
    st = o.add_stream("h264", rate=s.average_rate)
    st.width, st.height, st.pix_fmt = s.codec_context.width, s.codec_context.height, "yuv420p"
    st.bit_rate = bitrate
    st.options = {"preset": "slow", "maxrate": str(bitrate), "bufsize": str(2 * bitrate)}
    for f in ic.decode(video=0):
        for p in st.encode(av.VideoFrame.from_ndarray(f.to_ndarray(format="rgb24"), format="rgb24")):
            o.mux(p)
    for p in st.encode():
        o.mux(p)
    o.close(); ic.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="paper/video")
    ap.add_argument("--out", default="paper/multimedia")
    ap.add_argument("--scenarios", default="results/qual/scenarios.json")
    ap.add_argument("--zip-video", choices=["full", "small"], default="full",
                    help="video inside the zip: the original (RA-L zip limit 50 MB) or the <= 10 MB re-encode")
    a = ap.parse_args()
    import av, json
    src = os.path.join(a.video, "favor_supp.mp4")
    os.makedirs(a.out, exist_ok=True)
    with av.open(src) as c:
        s = c.streams.video[0]; w, h = s.codec_context.width, s.codec_context.height
        dur = float(c.duration / 1e6)
    scen = ""
    if os.path.exists(a.scenarios):
        lab = {"alphabet_soup": "Soup", "milk": "Milk", "bowl_ramekin": "Bowl-Ramekin", "bowl_stove": "Bowl-Stove"}
        for sc in json.load(open(a.scenarios)):
            scen += f"  - {lab.get(sc['task'], sc['task'])}, joint {sc['joint']} locked, seed {sc['seeds'][0]}" + \
                    (f" ({sc['note']})" if sc.get("note") else "") + "\n"
    open(os.path.join(a.out, "ReadMe.txt"), "w").write(README.format(w=w, h=h, dur=dur, scen=scen))
    open(os.path.join(a.out, "Summary.txt"), "w").write(SUMMARY)
    small = os.path.join(a.out, "favor_supp_10MB.mp4")
    reencode(src, small, 9.5)
    print(f"wrote {small} ({os.path.getsize(small) / 1e6:.1f} MB; single-video limit 10 MB)")
    z = os.path.join(a.out, "FAVOR_multimedia.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(src if a.zip_video == "full" else small, "favor_supp.mp4")
        zf.write(os.path.join(a.out, "ReadMe.txt"), "ReadMe.txt")
        zf.write(os.path.join(a.out, "Summary.txt"), "Summary.txt")
    print(f"wrote {z} ({os.path.getsize(z) / 1e6:.1f} MB, {a.zip_video} video; RA-L limit 50 MB)")


if __name__ == "__main__":
    main()
