#!/usr/bin/env python3
"""P5 animatic 草圖：每鏡一張 Illustrious（SDXL）低步數 txt2img → edit/drafts/sNNN.png（1280×720）。

~/creative/ComfyUI/comfyui/bin/python tools/p5_drafts.py [--only s010,s011] [--skip-existing] [--chunk 4]

取樣設定（只求看清構圖，不求品質）：euler_ancestral／normal、14 步、CFG 5.0（P3 正式出圖是 28 步、CFG 6.0）。
  1344×768（SDXL 的 16:9 bucket，與 p3_keyframe_test 相同）→ 縮到寬 1280 後置中裁成 1280×720。
  LoRA：luna_illustrious_v2 強度 0.8（同 p3_lora_gen），只有 prompt 含 lunacsf 才掛；其餘鏡頭用純 Illustrious。
  seed 固定為 20261100 + 鏡號，重跑同一 prompt 會得到同一張。
資料來源：shotlist.json 每鏡的 keyframe.first（與 meta.defaults.illustrious_negative；沒有就用 p3_lora_gen 的 NEG）。
  已有 P4／P6 正式關鍵幀 shots/sNNN/keyframe.png 的鏡頭直接用那張（縮放裁切到 1280×720），不重生。
  keyframe.first 是 null 的鏡頭（含 prompt 缺漏）跳過，animatic 會用灰底。
續跑：--skip-existing 跳過 edit/drafts 已有的檔案；紀錄在 edit/drafts/results.json（seed、prompt、步數、秒數）。
同一個 ComfyUI 請求內一次畫 --chunk 張（ComfyUI 用 --cache-none 啟動時每個請求都會重新載入模型，合併請求可省載入時間）。
重試：--only s010 --seed-bump 1 用 seed+1000 重生（--seed-bump 2 為 +2000），results.json 記最後採用的 seed 與 previous_seeds。
聯絡表：--sheet 產生 edit/drafts_sheet.jpg（每格標鏡號；生成完自動做，--sheet-only 只做聯絡表）。
ComfyUI 要先在 127.0.0.1:8188 跑：cd ~/creative/ComfyUI && comfyui/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-smart-memory --cache-none
"""
import argparse, io, json, re, sys, time, urllib.request, urllib.parse
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_lora_gen as L

ROOT = L.ROOT
HOST = L.HOST
LORA = "luna_illustrious_v2.safetensors"
LORA_STRENGTH = 0.8
TRIGGER = "lunacsf"
STEPS, CFG = 14, 5.0
GEN_W, GEN_H = 1344, 768
OUT_W, OUT_H = 1280, 720
OUT = ROOT / "edit/drafts"


def fit(img, w=OUT_W, h=OUT_H):
    """縮放涵蓋後置中裁切。"""
    img = img.convert("RGB")
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def with_quality(tags, quality=None):
    """prompt 沒寫畫風／品質詞就補 defaults.quality（沒有就用 P3 的 QUALITY）。"""
    low = tags.lower()
    return tags if "masterpiece" in low or "best quality" in low else tags.rstrip(", ") + (", " + quality if quality else L.QUALITY)


def graph(jobs):
    """jobs: [(key, positive, negative, seed, use_lora)] → 一個共用載入的 workflow；回傳 (wf, {key: save_node})。"""
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "Illustrious-XL-v1.0.safetensors"}}}
    use_lora = any(j[4] for j in jobs)
    if use_lora:
        wf["2"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": LORA,
                                                         "strength_model": LORA_STRENGTH, "strength_clip": LORA_STRENGTH}}
        wf["3"] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["2", 1], "stop_at_clip_layer": -2}}
    wf["3b"] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": -2}}
    saves = {}
    for i, (key, pos, neg, seed, lora) in enumerate(jobs):
        model, clip = (["2", 0], ["3", 0]) if lora else (["1", 0], ["3b", 0])
        n = f"j{i}_"
        wf[n + "p"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": pos}}
        wf[n + "n"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": neg}}
        wf[n + "l"] = {"class_type": "EmptyLatentImage", "inputs": {"width": GEN_W, "height": GEN_H, "batch_size": 1}}
        wf[n + "k"] = {"class_type": "KSampler", "inputs": {
            "model": model, "positive": [n + "p", 0], "negative": [n + "n", 0], "latent_image": [n + "l", 0], "seed": seed,
            "steps": STEPS, "cfg": CFG, "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0}}
        wf[n + "d"] = {"class_type": "VAEDecode", "inputs": {"samples": [n + "k", 0], "vae": ["1", 2]}}
        wf[n + "s"] = {"class_type": "SaveImage", "inputs": {"images": [n + "d", 0], "filename_prefix": "p5draft/x"}}
        saves[key] = n + "s"
    return wf, saves


def run(wf, saves):
    pid = L.api("/prompt", {"prompt": wf, "client_id": "p5drafts"})["prompt_id"]
    while True:
        hist = L.api("/history/" + pid)
        if pid in hist:
            st = hist[pid]["status"].get("status_str")
            if st == "error":
                raise RuntimeError(json.dumps(hist[pid]["status"], ensure_ascii=False)[:2000])
            if hist[pid].get("outputs") and all(s in hist[pid]["outputs"] for s in saves.values()):
                break
        time.sleep(1)
    res = {}
    for key, node in saves.items():
        im = hist[pid]["outputs"][node]["images"][0]
        res[key] = Image.open(io.BytesIO(urllib.request.urlopen(f"{HOST}/view?" + urllib.parse.urlencode(im)).read()))
    return res


def sheet(shots, out, dest, cols=8, cw=320, ch=180):
    """74 張拼成聯絡表（JPEG），每格左上標鏡號；沒有草圖的格子用灰底。"""
    from PIL import ImageDraw, ImageFont
    rows = -(-len(shots) // cols)
    img = Image.new("RGB", (cols * cw, rows * ch), (20, 20, 20))
    font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc", 22)
    for i, sid in enumerate(shots):
        p = out / f"{sid}.png"
        cell = Image.open(p).convert("RGB").resize((cw - 2, ch - 2), Image.LANCZOS) if p.exists() else Image.new("RGB", (cw - 2, ch - 2), (70, 70, 74))
        x, y = (i % cols) * cw, (i // cols) * ch
        img.paste(cell, (x + 1, y + 1))
        d = ImageDraw.Draw(img)
        d.rectangle((x + 1, y + 1, x + 64, y + 28), fill=(0, 0, 0))
        d.text((x + 6, y + 2), sid, font=font, fill=(255, 255, 80))
    img.save(dest, quality=88)
    print(f"聯絡表 {dest}（{len(shots)} 格，{img.width}x{img.height}）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shotlist", default="shotlist.json")
    ap.add_argument("--out", default=None, help="輸出資料夾（預設 edit/drafts）")
    ap.add_argument("--only", default="", help="逗號分隔鏡號，例如 s010,s011")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--chunk", type=int, default=4, help="一個 ComfyUI 請求畫幾張")
    ap.add_argument("--seed-bump", type=int, default=0, help="重試用：seed 加 1000×N")
    ap.add_argument("--sheet-only", action="store_true", help="只重做聯絡表")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    a = ap.parse_args()
    global STEPS, CFG
    STEPS, CFG = a.steps or STEPS, a.cfg or CFG
    out = Path(a.out) if a.out else OUT
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    sl = json.loads((ROOT / a.shotlist if not Path(a.shotlist).is_absolute() else Path(a.shotlist)).read_text())
    neg_default = (sl["meta"].get("defaults") or {}).get("illustrious_negative") or L.NEG
    quality = (sl["meta"].get("defaults") or {}).get("quality")
    only = set(filter(None, a.only.split(",")))
    log = out / "results.json"
    recs = {r["id"]: r for r in json.loads(log.read_text())} if log.exists() else {}

    sheet_dest = ROOT / "edit/drafts_sheet.jpg" if not a.out else out.parent / "drafts_sheet.jpg"
    if a.sheet_only:
        sheet([s["id"] for s in sl["shots"]], out, sheet_dest)
        return
    todo, copied, skipped, reuse = [], 0, [], []
    for s in sl["shots"]:
        sid = s["id"]
        if only and sid not in only:
            continue
        dest = out / f"{sid}.png"
        if a.skip_existing and dest.exists():
            continue
        kf = ROOT / "shots" / sid / "keyframe.png"
        if kf.exists():
            fit(Image.open(kf)).save(dest)
            recs[sid] = dict(id=sid, source=str(kf.relative_to(ROOT)))
            copied += 1
            print(f"{sid}: 使用正式關鍵幀 {kf.relative_to(ROOT)}", flush=True)
            continue
        tags = (s.get("keyframe") or {}).get("first")
        m = re.search(r"沿用\s*(s\d{3})", tags or "")
        if m and m.group(1) != sid:      # 例如「（沿用 s042 關鍵幀，推近 3% 的裁切）」：直接用那一鏡的草圖
            reuse.append((sid, m.group(1), "推近" in tags))
            continue
        if not tags:
            skipped.append(sid)
            continue
        pos = with_quality(tags, quality)
        seed = 20261100 + int(sid[1:]) + 1000 * a.seed_bump
        todo.append((sid, pos, neg_default, seed, TRIGGER in tags.lower()))
    def apply_reuse():
        for sid, ref, zoom in reuse:
            src = out / f"{ref}.png"
            if not src.exists():
                print(f"{sid}: 沿用 {ref} 但 {src.name} 不存在，跳過")
                continue
            img = Image.open(src).convert("RGB")
            if zoom:   # 推近約 3%：中心裁切再放大回原尺寸
                w, h = img.size
                cw, ch = round(w / 1.03), round(h / 1.03)
                img = img.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)).resize((w, h), Image.LANCZOS)
            img.save(out / f"{sid}.png")
            recs[sid] = dict(id=sid, source=f"reuse:{ref}" + ("+zoom1.03" if zoom else ""))
            print(f"{sid}: 沿用 {ref} 的草圖", flush=True)
        log.write_text(json.dumps(list(recs.values()), ensure_ascii=False, indent=1))

    log.write_text(json.dumps(list(recs.values()), ensure_ascii=False, indent=1))
    if skipped:
        print(f"沒有 keyframe.first、跳過（animatic 會用灰底）：{len(skipped)} 鏡 {', '.join(skipped[:10])}{' …' if len(skipped) > 10 else ''}")
    if not todo:
        apply_reuse()
        print(f"完成：複製正式關鍵幀 {copied} 張，沒有要生成的草圖")
        sheet([s["id"] for s in sl["shots"]], out, sheet_dest)
        return
    try:
        L.api("/system_stats")
    except Exception as e:
        sys.exit(f"連不上 ComfyUI（{HOST}）：{e}\n先啟動：cd ~/creative/ComfyUI && comfyui/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-smart-memory --cache-none")

    t_all = time.time()
    for i in range(0, len(todo), a.chunk):
        chunk = todo[i:i + a.chunk]
        t0 = time.time()
        wf, saves = graph(chunk)
        res = run(wf, saves)
        dt = time.time() - t0
        for key, pos, neg, seed, lora in chunk:
            fit(res[key]).save(out / f"{key}.png")
            prev = (recs.get(key) or {}).get("previous_seeds", []) + ([recs[key]["seed"]] if recs.get(key, {}).get("seed") not in (None, seed) else [])
            recs[key] = dict(id=key, seed=seed, previous_seeds=prev, lora=LORA if lora else None, lora_strength=LORA_STRENGTH if lora else None,
                             steps=STEPS, cfg=CFG, sampler="euler_ancestral", gen_size=f"{GEN_W}x{GEN_H}", prompt=pos, negative=neg,
                             seconds_per_image=round(dt / len(chunk), 1))
        log.write_text(json.dumps(list(recs.values()), ensure_ascii=False, indent=1))
        print(f"{', '.join(c[0] for c in chunk)}  {dt:.0f}s（{dt / len(chunk):.0f}s/張）  進度 {min(i + a.chunk, len(todo))}/{len(todo)}", flush=True)
    apply_reuse()
    print(f"完成：生成 {len(todo)} 張、複製關鍵幀 {copied} 張、跳過 {len(skipped)} 鏡，共 {time.time() - t_all:.0f}s → {out}")
    sheet([s["id"] for s in sl["shots"]], out, sheet_dest)


if __name__ == "__main__":
    main()
