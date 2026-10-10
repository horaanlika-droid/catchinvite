"""Extract the worn materials and hardware from the repository's IMG_1452.
Run from any directory. Requires Pillow, numpy and opencv-python-headless.
The reference's event copy is removed; the live invitation owns all event text.
"""
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'worn'
OUT.mkdir(exist_ok=True)
im = Image.open(ROOT / 'IMG_1452.png').convert('RGB').resize((724, 1560), Image.Resampling.LANCZOS)
im.save(OUT / 'chassis.webp', quality=86)

# Blend real paint samples into a large irregular surface. Mirrored/repeated
# small tiles make scratches look like wallpaper rather than accumulated wear.
rng = random.Random(22)
patch = im.crop((263, 350, 380, 412))
tile = Image.new('RGB', (1024, 1024), (23, 24, 22))
feather = Image.new('L', (58, 58))
ImageDraw.Draw(feather).ellipse((5, 5, 53, 53), fill=240)
feather = feather.filter(ImageFilter.GaussianBlur(7))
for _ in range(4300):
    x = rng.randrange(patch.width - 58)
    sample = patch.crop((x, 0, x + 58, 58)).rotate(rng.choice([0, 90, 180, 270]))
    tile.paste(sample, (rng.randrange(-30, 1024), rng.randrange(-30, 1024)), feather)
tile.save(OUT / 'paint.webp', quality=89)

# Preserve the original door, screws, glass and tape. Remove the reference's
# handwritten venue name from the paper, and its tiny venue name on the right.
deck = np.array(im.crop((24, 435, 703, 838)))
mask = np.zeros(deck.shape[:2], np.uint8)
cv2.fillPoly(mask, [np.array([(110, 56), (546, 29), (593, 84), (588, 121), (104, 151)])], 255)
dark = cv2.cvtColor(deck, cv2.COLOR_RGB2GRAY) < 92
mask = cv2.bitwise_and(mask, (dark * 255).astype(np.uint8))
mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
cv2.rectangle(mask, (557, 300), (642, 328), 255, -1)
deck = cv2.inpaint(deck, mask, 5, cv2.INPAINT_TELEA)
Image.fromarray(deck).save(OUT / 'deck.webp', quality=94)

# Only the actual hubs rotate: the photographed glass and tape remain fixed.
for name, cx, cy, radius in [('reel-a', 197, 238, 48), ('reel-b', 459, 238, 48)]:
    hub = Image.fromarray(deck).crop((cx-radius, cy-radius, cx+radius, cy+radius)).convert('RGBA')
    alpha = Image.new('L', hub.size)
    ImageDraw.Draw(alpha).ellipse((1, 1, 94, 94), fill=255)
    hub.putalpha(alpha.filter(ImageFilter.GaussianBlur(.65)))
    hub.save(OUT / f'{name}.webp', quality=94)

# Actual worn button faces; blank their icons so live SVGs stay legible.
for name, box, icon in [
    ('rew', (105, 888, 243, 1026), (43, 53, 96, 99)),
    ('play', (240, 858, 479, 1048), (88, 57, 162, 126)),
    ('stop', (486, 888, 632, 1026), (38, 51, 105, 101)),
]:
    key = np.array(im.crop(box))
    keymask = np.zeros(key.shape[:2], np.uint8)
    cv2.rectangle(keymask, icon[:2], icon[2:], 255, -1)
    key = cv2.inpaint(key, keymask, 5, cv2.INPAINT_TELEA)
    Image.fromarray(key).save(OUT / f'key-{name}.webp', quality=93)

# Worn versions of the repository's stickers, with their chipped paper edges.
polygons = {
    'lastcall': [(17,87),(206,60),(237,301),(26,329),(0,125)],
    'love22': [(548,65),(695,62),(720,91),(711,236),(686,252),(534,231)],
    'banana': [(267,294),(309,241),(326,160),(348,145),(381,157),(405,214),(377,272),(301,313),(278,312)],
    'year22': [(388,253),(457,208),(495,218),(550,248),(548,290),(513,319),(478,325),(476,345),(500,365),(485,405),(429,389),(386,356),(397,328),(418,327),(409,312),(390,296)],
    'blond': [(27,352),(240,333),(256,389),(35,422)],
    'flashing': [(528,365),(702,342),(716,420),(528,448),(510,392)],
    'imcatch': [(27,1010),(104,1009),(137,1052),(184,1066),(225,1117),(215,1159),(156,1194),(27,1177),(0,1130),(10,1066)],
    'disco': [(11,1237),(51,1201),(125,1180),(169,1180),(186,1212),(166,1243),(181,1275),(157,1321),(45,1332),(8,1295)],
    'kanye': [(577,1031),(628,1024),(701,1046),(724,1090),(708,1172),(693,1209),(647,1256),(587,1246),(539,1220),(524,1182),(543,1094)],
}
for name, points in polygons.items():
    alpha = Image.new('L', im.size)
    ImageDraw.Draw(alpha).polygon(points, fill=255)
    alpha = alpha.filter(ImageFilter.GaussianBlur(.6))
    sticker = im.convert('RGBA')
    sticker.putalpha(alpha)
    sticker.crop(alpha.getbbox()).save(OUT / f'{name}.webp', quality=92)

# A subtle transparent layer of accumulated fine scratches for the page surface.
rng = random.Random(1452)
wear = Image.new('RGBA', (900, 1000))
draw = ImageDraw.Draw(wear)
for _ in range(600):
    x, y = rng.randrange(900), rng.randrange(1000)
    length = rng.randrange(2, 30)
    dx, dy = length, rng.randrange(-17, 6)
    shade = rng.randrange(135, 205)
    draw.line([(x, y), (x + dx * .55, y + dy * .5), (x + dx, y + dy)], fill=(shade, shade-7, shade-19, rng.randrange(15, 85)), width=1)
wear.save(OUT / 'scratches.webp', lossless=True)
# Paper stays a separate, lightweight tile so the invitation remains live text.
paper_rng = random.Random(22)
paper = Image.new('RGB', (256, 256))
paper.putdata([(v, v - 9, v - 30) for v in (paper_rng.randrange(225, 246) for _ in range(256 * 256))])
paper.save(ROOT / 'assets' / 'paper.jpg', quality=80, optimize=True)
print('Extracted reference materials to', OUT)
