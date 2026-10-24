#!/usr/bin/env python3
"""P3 步驟 4–5：用正式管線（Illustrious-XL ＋ LUNA LoRA）出圖。

python3 tools/p3_lora_gen.py --lora luna_illustrious_v1.safetensors --set eval --out lora/v1/eval [--strength 0.8] [--seeds 1]
python3 tools/p3_lora_gen.py --lora luna_illustrious_v2.safetensors --set style --out refs/style_samples

LoRA 檔要先放到 ComfyUI 的 models/loras/（用符號連結）。經 ComfyUI HTTP API（127.0.0.1:8188）。
--set eval：30 個 v1 評估／擴充資料用的 prompt；--set style：每個主要段落 1–2 張風格樣張（style bible 的場景與色調）。
"""
import argparse, json, sys, time, urllib.request, urllib.parse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_prepare_dataset import outfit_tags

HOST = "http://127.0.0.1:8188"
ROOT = Path(__file__).resolve().parent.parent
# v2 評估：服裝標籤全寫會把特寫拉成膝上構圖，所以和訓練標籤一樣依構圖只寫看得到的（outfit_tags）；
# 瞳色沒進訓練標籤，底模會畫成綠色，固定寫 grey eyes（style bible：深紫灰）
BASE = "lunacsf, 1girl, solo, grey eyes, dark purple hair, "


def prompt_outfit(tags):
    """outfit_tags 之外，補上底模容易畫錯色的部分：特寫的高領會變紫或黑、腰以上鏡頭露出的下半身會變黑褲。"""
    o = outfit_tags(tags, "visible")
    if tags.startswith(("close-up", "portrait")):
        o += "white shirt, sleeveless, "
    elif tags.startswith("upper body"):
        o += "white skirt, "
    return o
QUALITY = ", masterpiece, best quality, anime screencap, clean lineart, cel shading"
NEG = ("lowres, bad anatomy, bad hands, extra fingers, missing fingers, fused fingers, extra digits, text, watermark, signature, logo, "
       "letters, username, green eyes, aqua eyes, blue eyes, two thigh straps, thighhighs, kneehighs, jacket, hat, glasses, chibi, 3d, realistic, photo, blurry, multiple girls")

EVAL = [
    ("front_stand", 832, 1216, "full body, standing, looking at viewer, white background, simple background"),
    ("back", 832, 1216, "full body, from behind, white background, simple background"),
    ("side_right", 832, 1216, "full body, from side, profile, standing, white background, simple background"),
    ("3q_smile", 896, 1152, "cowboy shot, three-quarter view, smile, white background, simple background"),
    ("portrait", 1024, 1024, "portrait, looking at viewer, determined, white background, simple background"),
    ("face_close", 1024, 1024, "close-up, face, looking at viewer, earrings, white background, simple background"),
    ("upper_smile", 896, 1152, "upper body, smile, looking at viewer, simple background"),
    ("run", 1216, 832, "full body, running, from side, motion lines, white background, simple background"),
    ("jump", 832, 1216, "full body, jumping, midair, excited, simple background"),
    ("kneel_land", 1216, 832, "full body, one knee, hand on ground, looking ahead, rooftop, night"),
    ("arms_open", 832, 1216, "full body, outstretched arms, smile, white background, simple background"),
    ("point", 832, 1216, "cowboy shot, pointing at viewer, smile, white background, simple background"),
    ("fist_up", 832, 1216, "cowboy shot, arm up, clenched hand, open mouth smile, night, city lights, bokeh"),
    ("look_up", 896, 1152, "upper body, looking up, parted lips, dark, light from above"),
    ("hand_chest", 896, 1152, "upper body, hand on own chest, looking at hand, simple background"),
    ("night_street", 832, 1216, "full body, walking, night, street, wet, rain"),
    ("rooftop", 832, 1216, "full body, standing, rooftop, night, city lights"),
    ("control_room", 896, 1152, "upper body, hands on table, indoors, dark, cyan light"),
    ("walkway", 832, 1216, "full body, standing, railing, looking afar, night, city lights, from above"),
    ("dawn_back", 832, 1216, "full body, from behind, looking afar, sunrise, horizon"),
    ("rain_portrait", 1024, 1024, "portrait, looking at viewer, rain, night"),
    ("rim_light", 896, 1152, "upper body, from side, profile, night sky, rim lighting"),
    ("eyes_closed", 1024, 1024, "close-up, face, closed eyes, dark background, light from below"),
    ("sitting", 1216, 832, "full body, sitting, knees up, smile, white background, simple background"),
    ("back_hair", 896, 1152, "from behind, upper body, hair, braid, hair chain, white background, simple background"),
    ("lean_sprint", 832, 1216, "full body, leaning forward, ready to run, determined, white background, simple background"),
    ("laugh", 896, 1152, "upper body, laughing, backlighting, sunrise"),
    ("serious_3q", 896, 1152, "cowboy shot, serious, three-quarter view, dark background, cyan light"),
    ("crouch", 1216, 832, "full body, crouching, hand on ground, looking up, indoors, dark, cyan light"),
    ("walk_front", 832, 1216, "full body, walking towards viewer, night, street, wet"),
]
STYLE = [
    ("intro_pod", 1216, 832, "full body, standing, closed eyes, inside glass capsule, dark round room, monitor wall, cyan light lines on floor, light from below, cold blue-black"),
    ("hook_rooftop", 1216, 832, "full body, one knee, landing, rooftop, night, rain, grey city in distance, colorful district far away, wind"),
    ("verse_street", 1216, 832, "full body, standing in puddle, looking at own hand, surprised, night, shopping street, wet stone pavement, greyscale background, neutral grey light"),
    ("prechorus_cars", 1216, 832, "full body, jumping, midair, hover car, night, street, motion blur, dynamic angle"),
    ("chorus1_burst", 1216, 832, "full body, outstretched arms, floating, colorful light burst from chest, neon city, rain, glowing"),
    ("chorus1_climb", 832, 1216, "full body, jumping, climbing black metal tower, cyan glowing lines, magenta light bands, night sky, from below"),
    ("bridge_console", 1216, 832, "upper body, hands on round console, cyan glowing rings, dark control room, smile, light from below"),
    ("build_column", 832, 1216, "full body, standing on console, both hands on glowing glass pillar, dark room, looking up, determined"),
    ("final_parapet", 832, 1216, "full body, standing on railing, arm up, clenched hand, smile, night, colorful city lights below, from below"),
    ("outro_lookback", 1216, 832, "upper body, looking back, looking at viewer, monitor wall, colorful screens, dark room, slight smile"),
]


def api(path, data=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def workflow(lora, strength, pos, w, h, seed, steps, cfg):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "Illustrious-XL-v1.0.safetensors"}},
        "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": lora,
                                                     "strength_model": strength, "strength_clip": strength}},
        "3": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["2", 1], "stop_at_clip_layer": -2}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": pos}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": NEG}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                                                   "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler_ancestral",
                                                   "scheduler": "normal", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "p3lora/x"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--set", choices=["eval", "style"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=6.0)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    log = out / "results.json"
    recs = json.loads(log.read_text()) if log.exists() else []
    specs = EVAL if a.set == "eval" else STYLE
    for k in range(a.seeds):
        for i, (name, w, h, tags) in enumerate(specs, 1):
            seed = 20261100 + i + 1000 * k
            dest = out / f"{i:02d}_{name}_s{k}.png"
            if dest.exists():
                continue
            pos = BASE + prompt_outfit(tags) + tags + QUALITY
            t0 = time.time()
            pid = api("/prompt", {"prompt": workflow(a.lora, a.strength, pos, w, h, seed, a.steps, a.cfg), "client_id": "p3lora"})["prompt_id"]
            while True:
                hist = api("/history/" + pid)
                if pid in hist and hist[pid].get("outputs"):
                    break
                if pid in hist and hist[pid]["status"].get("status_str") == "error":
                    raise RuntimeError(json.dumps(hist[pid]["status"])[:2000])
                time.sleep(1)
            im = hist[pid]["outputs"]["9"]["images"][0]
            dest.write_bytes(urllib.request.urlopen(f"{HOST}/view?" + urllib.parse.urlencode(im)).read())
            recs.append(dict(name=name, file=str(dest.relative_to(ROOT)), seed=seed, lora=a.lora, strength=a.strength,
                             steps=a.steps, cfg=a.cfg, size=f"{w}x{h}", prompt=pos, negative=NEG, seconds=round(time.time() - t0, 1)))
            log.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
            print(name, recs[-1]["seconds"], flush=True)


if __name__ == "__main__":
    main()
