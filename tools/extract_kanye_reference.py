"""Extract the Kanye portrait sticker from IMG_1452.png.

Run from any directory with Pillow installed. This replaces only the portrait
asset; the rest of the invitation artwork stays untouched.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SIZE = (724, 1560)
CROP_BOX = (524, 1024, 724, 1256)
# Silhouette of the torn sticker in the lower-right of IMG_1452, relative to
# CROP_BOX. The mask keeps the real paper edge while dropping the device body.
STICKER_EDGE = [
    (53, 7), (104, 0), (177, 22), (200, 66), (184, 148), (169, 185),
    (123, 232), (63, 222), (15, 196), (0, 158), (19, 70),
]

reference = Image.open(ROOT / "IMG_1452.png").convert("RGB")
reference = reference.resize(REFERENCE_SIZE, Image.Resampling.LANCZOS)
sticker = reference.crop(CROP_BOX).convert("RGBA")
mask = Image.new("L", sticker.size, 0)
ImageDraw.Draw(mask).polygon(STICKER_EDGE, fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(0.55))
sticker.putalpha(mask)
output = ROOT / "assets" / "stickers" / "kanye-1452.png"
sticker.save(output, optimize=True)
print(f"Wrote {output}")
