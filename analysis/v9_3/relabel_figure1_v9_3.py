"""v9.3 — Figure 1 with reader-facing panel titles.

The original Figure 1 (v9/manuscript/figures/Figure1_segmentation.png) is a finished 3-panel screenshot from a 3D
viewer with internal titles ("DESS (GT)", "PD_dense (4x)", "PD_orig (2.9mm)") burned into the image; no generating
script survives. This script blanks the title strip of each panel (top 33 px; the structure legend starts below it)
and writes reader-facing titles. Rendering pixels are untouched.

Output: ../manuscript/figures/Figure1_segmentation.png (v9.3 copy; the v9 original is not modified)
Usage: python relabel_figure1_v9_3.py [--dry_run]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "v9" / "manuscript" / "figures" / "Figure1_segmentation.png"
OUT = HERE.parent / "manuscript" / "figures" / "Figure1_segmentation.png"
PANEL_X = [0, 595, 1190]                     # left borders of the three panels (1785 x 600 image)
TITLES = ["DESS, in-house segmentation (same knee)",
          "PD/IW, automated + depth densification",
          "PD/IW, automated (original 3-mm sections)"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry_run", action="store_true"); a = ap.parse_args()
    im = Image.open(SRC).convert("RGB")
    assert im.size == (1785, 600), im.size
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 21)
    except OSError:
        font = ImageFont.load_default()
    for x0, t in zip(PANEL_X, TITLES):
        d.rectangle([x0 + 2, 2, x0 + 592, 33], fill="white")          # keep the 1-px panel borders
        d.text((x0 + 6, 5), t, fill="black", font=font)
    if a.dry_run:
        print("[dry_run] not saved"); return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    im.save(OUT)
    print("[ok]", OUT)


if __name__ == "__main__":
    main()
