# -*- coding: utf-8 -*-
"""Repair the reel sprite: the generated source clipped the brown tape pack
with flat vertical cuts on the left/right, so on the page the reels read as
"cut at the edges". The tape winding is concentric, therefore the missing
side caps of the pack are reconstructed from the radial colour profile of the
intact sectors (top/bottom), plus grain matched to the local variance.
Run from any directory. Requires Pillow and numpy.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'assets' / 'layers' / 'reel.webp'

im = Image.open(SRC).convert('RGBA')
a = np.asarray(im).astype(np.float32)
rgb, alpha = a[..., :3], a[..., 3]
H, W = alpha.shape

R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
brown = (R - B > 25) & (R > 50) & (alpha > 200)
ys, xs = np.nonzero(brown)
cx, cy = xs.mean(), ys.mean()

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
r_int = np.clip(np.round(rr).astype(int), 0, min(H, W) // 2 - 1)

# intact sectors of the pack: away from the flat cuts
intact = brown & (np.abs(xx - cx) < 100) & (rr < 127)
mean = np.zeros((r_int.max() + 1, 3), np.float32)
std = np.ones((r_int.max() + 1, 3), np.float32)
for r in range(mean.shape[0]):
    m = intact & (r_int == r)
    if m.sum() > 8:
        mean[r] = rgb[m].mean(axis=0)
        std[r] = rgb[m].std(axis=0) + 1.0
# fill sparse radii by neighbour interpolation
for r in range(1, mean.shape[0] - 1):
    if mean[r].sum() == 0:
        mean[r] = (mean[r - 1] + mean[r + 1]) / 2
        std[r] = (std[r - 1] + std[r + 1]) / 2

# missing caps: inside the pack circle, beyond the vertical cut lines;
# only where the pixel is still the dark lobe (keeps the script idempotent)
cut = 116.0
lum = rgb.mean(axis=2)
dark = np.clip((70 - lum) / 20.0, 0, 1)
w = np.clip((np.abs(xx - cx) - (cut - 2)) / 4.0, 0, 1) * (rr < 127.5) * dark
rng = np.random.default_rng(7)
noise = rng.normal(0, 1, (H, W, 3)).astype(np.float32)
noise = np.asarray(Image.fromarray(
    ((noise - noise.min()) / (float(np.ptp(noise)) + 1e-6) * 255).astype(np.uint8)
).filter(ImageFilter.GaussianBlur(0.7)), np.float32)
noise = (noise - noise.mean()) / (noise.std() + 1e-6)
patch = mean[r_int] + noise * (std[r_int] * 0.55)[..., :3]
patch = np.clip(patch, 0, 255)

w3 = w[..., None]
out = rgb * (1 - w3) + patch * w3
res = np.dstack([np.clip(out, 0, 255), alpha]).astype(np.uint8)
Image.fromarray(res, 'RGBA').save(SRC, quality=90)
print('reel repaired:', SRC, im.size)
