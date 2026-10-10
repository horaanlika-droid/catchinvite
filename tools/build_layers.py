# -*- coding: utf-8 -*-
"""Post-process the AI-generated layers in gen/raw/ into webp assets.

The generated plate keeps the reference's dark grungy look but is a clean
layer: transparent outside the rounded body, measured reel hubs so the live
HTML can position the rotating sprites exactly. Run from any directory.
Requires Pillow and numpy.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'gen' / 'raw'
OUT = ROOT / 'assets' / 'layers'
OUT.mkdir(parents=True, exist_ok=True)


def luminance(im):
    return np.asarray(im.convert('L')).astype(np.float32)


def alpha_from_luma(im, lo=10, hi=22):
    """Transparent where the render background is pure black."""
    l = luminance(im)
    a = np.clip((l - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return a


def bbox_of(alpha, thr=16):
    ys, xs = np.nonzero(alpha > thr)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


# ---------- plate: body layer + measured reel hubs ----------
plate = Image.open(RAW / 'plate.png').convert('RGB')
a = alpha_from_luma(plate)
x0, y0, x1, y1 = bbox_of(a)
plate = plate.crop((x0, y0, x1, y1))
alpha = Image.fromarray(alpha_from_luma(plate)).filter(ImageFilter.GaussianBlur(.6))
plate.putalpha(alpha)
W, H = plate.size

# Reel hub centres/radius, in % of the plate, verified against a debug
# composite overlay (gen/debug_overlay.png): circles drawn at these values
# sit exactly on the generated hubs. The overlay sprite is a touch larger
# than the painted hub so rotation reads cleanly.
reels = [
    {'cx': 30.7, 'cy': 42.9, 'r': 8.4},
    {'cx': 68.6, 'cy': 42.9, 'r': 8.4},
]
plate.save(OUT / 'plate.webp', quality=88)

# ---------- reel sprite: circular cut ----------
reel = Image.open(RAW / 'reel.png').convert('RGB')
ra = alpha_from_luma(reel, 14, 30)
rx0, ry0, rx1, ry1 = bbox_of(ra)
reel = reel.crop((rx0, ry0, rx1, ry1))
side = max(reel.size)
sq = Image.new('RGB', (side, side), (0, 0, 0))
sq.paste(reel, ((side - reel.width) // 2, (side - reel.height) // 2))
sq = sq.resize((360, 360), Image.Resampling.LANCZOS)
mask = Image.new('L', (360, 360))
from PIL import ImageDraw
ImageDraw.Draw(mask).ellipse((2, 2, 358, 358), fill=255)
sq.putalpha(mask.filter(ImageFilter.GaussianBlur(1)))
sq.save(OUT / 'reel.webp', quality=90)

# ---------- key caps ----------
for src, name in [('key-small.png', 'key-small.webp'), ('key-play.png', 'key-play.webp')]:
    key = Image.open(RAW / src).convert('RGB')
    ka = alpha_from_luma(key, 8, 20)
    kx0, ky0, kx1, ky1 = bbox_of(ka)
    key = key.crop((kx0, ky0, kx1, ky1))
    key.putalpha(Image.fromarray(alpha_from_luma(key, 6, 18)).filter(ImageFilter.GaussianBlur(.5)))
    key.save(OUT / name, quality=90)

# ---------- body texture ----------
tex = Image.open(RAW / 'body-tex.png').convert('RGB').resize((1024, 559), Image.Resampling.LANCZOS)
tex.save(OUT / 'body-tex.webp', quality=80)

info = {'plate': {'w': W, 'h': H}, 'reels': reels}
(ROOT / 'gen' / 'layers.json').write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
