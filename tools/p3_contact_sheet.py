#!/usr/bin/env python3
"""把一批圖片排成帶編號的對照表，方便目視挑圖。python3 tools/p3_contact_sheet.py OUT.png IMG...（用 ComfyUI 的 python，需 PIL）"""
import sys
from PIL import Image, ImageDraw
out, files = sys.argv[1], sys.argv[2:]
H, cols = 420, 4
cells = []
for f in files:
    im = Image.open(f).convert("RGB")
    im = im.resize((int(im.width * H / im.height), H))
    cells.append((f, im))
W = max(c[1].width for c in cells)
rows = (len(cells) + cols - 1) // cols
sheet = Image.new("RGB", (W * cols, (H + 22) * rows), "white")
d = ImageDraw.Draw(sheet)
for i, (f, im) in enumerate(cells):
    x, y = (i % cols) * W, (i // cols) * (H + 22)
    sheet.paste(im, (x, y + 22))
    d.text((x + 4, y + 4), f.split("/")[-1][:40], fill="black")
sheet.save(out)
