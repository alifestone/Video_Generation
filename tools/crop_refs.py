#!/usr/bin/env python3
"""依 refs/crops.json 從角色表裁切單格，輸出 refs/crops/<name>.png（可重複執行）。"""
import json, sys
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent.parent
sheet = root / "refs" / "動畫_女生_LUNA.jpg"
cfg = root / "refs" / "crops.json"
out = root / "refs" / "crops"

def main():
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(sheet).convert("RGB")
    for c in json.loads(cfg.read_text(encoding="utf-8")):
        x0, y0, x1, y1 = c["box"]
        assert 0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height, c
        img.crop((x0, y0, x1, y1)).save(out / f"{c['name']}.png")
        print("wrote", out / f"{c['name']}.png", (x1 - x0, y1 - y0))

if __name__ == "__main__":
    sys.exit(main())
