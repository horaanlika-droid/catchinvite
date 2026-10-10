# -*- coding: utf-8 -*-
"""Build the static device plate (holes punched for reels) from gen/device_raw.png.
Crop box in source px: x 360..1050, y 36..726  -> 690x690 square.
Reel centres / radius measured from dark-disc transitions in the source."""
import numpy as np
from PIL import Image
import os, json

SRC = 'gen/device_raw.png'
CX0, CY0, S = 360, 36, 690
REELS_SRC = [(579.0, 308.0), (838.0, 308.0)]   # source px
R_HOLE = 124.0                                  # hole radius, px (crop scale 1:1)

dev = Image.open(SRC).convert('RGBA').crop((CX0, CY0, CX0 + S, CY0 + S))
arr = np.asarray(dev).astype(np.float64)
YY, XX = np.mgrid[0:S, 0:S]
alpha = np.ones((S, S))
reels = [(x - CX0, y - CY0) for x, y in REELS_SRC]
for rx, ry in reels:
    d = np.sqrt((XX - rx) ** 2 + (YY - ry) ** 2)
    alpha *= np.clip((d - (R_HOLE - 1.5)) / 2.0, 0, 1)     # anti-aliased hole
    ring = np.clip((R_HOLE + 12 - d) / 12, 0, 1) ** 1.5     # recess shadow
    for c in range(3):
        arr[..., c] *= (1 - 0.7 * ring)
arr[..., 3] = alpha * 255
Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGBA').save('assets/player.png', optimize=True)

info = {
  'size': S,
  'reels': [{'cx': x / S * 100, 'cy': y / S * 100, 'r': R_HOLE / S * 100} for x, y in reels],
}
json.dump(info, open('gen/layout.json', 'w'), indent=1)
print(json.dumps(info, indent=1))
