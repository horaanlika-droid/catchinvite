# -*- coding: utf-8 -*-
"""Give the device plate exactly THREE physical keys.
1. Overwrite the key strip (y 546..594) with the same-width brushed-metal strip
   copied from directly above it (y 486..534) -- keeps the streak texture.
2. Draw three bevelled keys centred on the row y=552.
Reel holes (alpha) are left untouched."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PATH = 'assets/player.png'
S = 690
KEY_Y, KEY_H, KEY_W = 552, 32, 73
NEW_X = [216, 309, 402]
STRIP_Y0, STRIP_Y1 = 546, 594
SRC_Y0 = STRIP_Y0 - 60

img = Image.open(PATH).convert('RGBA')
arr = np.asarray(img).copy()
alpha = arr[..., 3].copy()

# 1. metal strip copy (x 60..630 covers all old keys)
X0, X1 = 60, 630
arr[STRIP_Y0:STRIP_Y1, X0:X1, :3] = arr[SRC_Y0:SRC_Y0 + (STRIP_Y1 - STRIP_Y0), X0:X1, :3]
arr[..., 3] = alpha

base = Image.fromarray(arr, 'RGBA')

# 2. keys
def key_sprite(w, h):
    s = Image.new('RGBA', (w + 16, h + 16), (0, 0, 0, 0))
    shadow = Image.new('RGBA', s.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((6, 9, w + 6, h + 9), radius=5, fill=(0, 0, 0, 170))
    s = Image.alpha_composite(s, shadow.filter(ImageFilter.GaussianBlur(3)))
    grad = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        t = y / max(1, h - 1)
        c = int(44 * (1 - t) + 14 * t)
        grad[y, :, :3] = (c, c - 1, c - 3)
        grad[y, :, 3] = 255
    cap = Image.fromarray(grad, 'RGBA')
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=5, fill=255)
    cap.putalpha(m)
    hl = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(hl).rounded_rectangle((2, 1, w - 3, 3), radius=2, fill=(255, 255, 255, 40))
    cap = Image.alpha_composite(cap, hl)
    s.alpha_composite(cap, (8, 8))
    return s

sprite = key_sprite(KEY_W, KEY_H)
for x in NEW_X:
    base.alpha_composite(sprite, (x - 8, KEY_Y - 8))

out = np.asarray(base).copy()
out[..., 3] = alpha          # restore reel holes exactly
Image.fromarray(out, 'RGBA').save(PATH, optimize=True)
print('keys at x =', NEW_X)
