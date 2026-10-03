"""
README previews: paper/video/<scenario>_2x2.mp4 -> paper/video/<scenario>_preview.gif
(2x playback speed, 420 px wide, ~3 MB each). Needs av + pillow (LIBERO container).
Usage: python scripts_paper/make_preview_gif.py [--video paper/video]
"""
import argparse, glob, os

import av
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="paper/video")
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--every", type=int, default=6, help="keep every n-th frame of the 20 fps video")
    ap.add_argument("--ms", type=int, default=150, help="display time per kept frame")
    a = ap.parse_args()
    for mp4 in sorted(glob.glob(os.path.join(a.video, "*_2x2.mp4"))):
        frames = []
        for i, fr in enumerate(av.open(mp4).decode(video=0)):
            if i % a.every == 0:
                im = fr.to_image()
                frames.append(im.resize((a.width, round(im.height * a.width / im.width)), Image.LANCZOS))
        pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
        out = mp4.replace("_2x2.mp4", "_preview.gif")
        pal[0].save(out, save_all=True, append_images=pal[1:], duration=a.ms, loop=0, optimize=True)
        print(f"wrote {out} ({len(pal)} frames, {os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
