#!/usr/bin/env python3
"""把 refs/experiment 的 A|B 結果拼成 compare.png（列=目標，欄=A|B），附標籤。"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
root = Path(__file__).resolve().parent.parent / "refs/experiment"
recs = json.loads((root / "results.json").read_text())["records"]
tg = []
for r in recs:
    if r["target"] not in tg: tg.append(r["target"])
H = 640; LBL = 36
try: font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 22)
except Exception: font = ImageFont.load_default()
cells = {}
for r in recs:
    im = Image.open(root.parent.parent / r["file"]).convert("RGB")
    cells[(r["target"], r["mode"])] = im.resize((int(im.width * H / im.height), H), Image.LANCZOS)
colw = 540
W = colw * 2 + 160
sheet = Image.new("RGB", (W, LBL + (H + LBL) * len(tg)), "white"); d = ImageDraw.Draw(sheet)
d.text((160 + colw // 2 - 90, 6), "A: sheet only", fill="black", font=font)
d.text((160 + colw + colw // 2 - 110, 6), "B: sheet + crop", fill="black", font=font)
for i, t in enumerate(tg):
    y = LBL + i * (H + LBL)
    d.text((8, y + H // 2), f"{i+1:02d} {t}", fill="black", font=font)
    for j, m in enumerate("AB"):
        c = cells.get((t, m))
        if c: sheet.paste(c, (160 + j * colw + (colw - c.width) // 2, y))
sheet.save(root / "compare.png"); print(sheet.size)
