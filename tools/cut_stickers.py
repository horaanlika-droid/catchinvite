# -*- coding: utf-8 -*-
"""Cut individual stickers out of the sheet photo IMG_1448.jpeg with GrabCut.
Boxes are in original-photo pixel coordinates (1024x1280)."""
import cv2, numpy as np, os
from PIL import Image

SRC = 'IMG_1448.jpeg'
OUT = 'assets/stickers'
os.makedirs(OUT, exist_ok=True)

BOXES = {
    'asv':        (320, 230, 435, 296),
    'banana':     (474, 256, 542, 350),
    'love22':     (566, 262, 692, 372),
    'lastcall':   (243, 302, 432, 462),
    'year22':     (418, 358, 568, 502),
    'blond':      (230, 452, 372, 503),
    'flashing':   (535, 460, 688, 522),
    'imcatch':    (200, 523, 364, 632),
    'disco':      (388, 508, 522, 602),
    'kanye':      (528, 533, 664, 667),
}
PAD = 6

img = cv2.imread(SRC)
H, W = img.shape[:2]

def cut(name, box):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - PAD), max(0, y0 - PAD)
    x1, y1 = min(W, x1 + PAD), min(H, y1 + PAD)
    crop = img[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (PAD // 2, PAD // 2, crop.shape[1] - PAD, crop.shape[0] - PAD)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    # keep largest connected component (the sticker itself)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    # fill holes, then smooth edge
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    filled = cv2.GaussianBlur(filled, (5, 5), 1.2)
    alpha = np.clip((filled.astype(np.float32) - 110) / 60 * 255, 0, 255).astype(np.uint8)
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = alpha
    ys, xs = np.nonzero(alpha > 8)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    path = f'{OUT}/{name}.png'
    cv2.imwrite(path, rgba)
    print(name, rgba.shape[1], 'x', rgba.shape[0], 'coverage %.2f' % (alpha > 128).mean())
    return path

for k, b in BOXES.items():
    cut(k, b)

# contact sheet for visual check
paths = [f'{OUT}/{k}.png' for k in BOXES]
tiles = []
for p in paths:
    t = Image.open(p).convert('RGBA')
    t.thumbnail((180, 180))
    bg = Image.new('RGBA', (190, 190), (40, 40, 40, 255))
    bg.paste(t, ((190 - t.width) // 2, (190 - t.height) // 2), t)
    tiles.append(bg)
cols = 5
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new('RGBA', (cols * 190, rows * 190), (40, 40, 40, 255))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 190, (i // cols) * 190))
sheet.convert('RGB').save('gen/stickers_contact.png')
print('contact sheet ok')
