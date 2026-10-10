# -*- coding: utf-8 -*-
"""Cut the torn "НАМ 1 ГОД!" paper sticker out of the second reference photo
IMG_1453.png (silver Walkman) with GrabCut, same pipeline as cut_stickers.py.
Box is in original-photo pixel coordinates (1254x1254).
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'IMG_1453.png'
OUT = ROOT / 'assets' / 'stickers' / 'nam1god.png'

x0, y0, x1, y1 = 650, 380, 866, 666
PAD = 8

img = cv2.imread(str(SRC))
crop = img[y0 - PAD:y1 + PAD, x0 - PAD:x1 + PAD].copy()
mask = np.zeros(crop.shape[:2], np.uint8)
rect = (PAD // 2, PAD // 2, crop.shape[1] - PAD, crop.shape[0] - PAD)
bgd = np.zeros((1, 65), np.float64)
fgd = np.zeros((1, 65), np.float64)
cv2.grabCut(crop, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

# drop the yellow smiley sticker that touches the top-right corner
b, g, r = crop[..., 0].astype(int), crop[..., 1].astype(int), crop[..., 2].astype(int)
yellow = (r - b > 55) & (g - b > 35) & (r > 140)
m[yellow] = 0

n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
if n > 1:
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = np.where(lab == big, 255, 0).astype(np.uint8)
cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
filled = np.zeros_like(m)
cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
filled = cv2.GaussianBlur(filled, (5, 5), 1.2)
alpha = np.clip((filled.astype(np.float32) - 110) / 60 * 255, 0, 255).astype(np.uint8)

rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
rgba[..., 3] = alpha
ys, xs = np.nonzero(alpha > 8)
rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), rgba)
print('nam1god:', rgba.shape[1], 'x', rgba.shape[0], 'coverage %.2f' % (alpha > 128).mean())
