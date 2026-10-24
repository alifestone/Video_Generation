#!/usr/bin/env python3
"""G0 第 2 題：驗證正式關鍵幀的合成流程（s017、s024）。

~/creative/ComfyUI/comfyui/bin/python tools/p3_keyframe_test.py [--only s017]

A. Qwen-Image-Edit-2511（Lightning 4 步）：image1＝角色表、image2/3＝物件參考圖（refs/entities/），依鏡頭首幀合成構圖。
B. Illustrious-XL＋LUNA LoRA img2img：把 A 的角色與畫風統一成 LoRA 的樣子。兩種做法並排比較：
   B1 只用 img2img（denoise 0.45）；B2 加 canny ControlNet（用 A 的邊緣固定構圖，denoise 0.6）。
C. 後製：灰化（飽和度 0、亮度不變，style bible）與 @broadcast_wave 光環示意（glossary：由後製合成）。
輸出到 refs/keyframe_test/<shot>/，prompt 與參數記在 results.json。
"""
import argparse, json, shutil, sys, time, urllib.request, urllib.parse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_lora_gen as L

HOST = L.HOST
ROOT = L.ROOT
COMFY_INPUT = Path.home() / "creative/ComfyUI/input"
W, H = 1344, 768   # SDXL 的 16:9 附近 bucket；正式片 1280×720 由這個尺寸縮放裁切
LORA = "luna_illustrious_v2.safetensors"
CHAR = ("Keep the exact same character and outfit as the reference sheet in image 1: dark purple high ponytail with braids on the crown and a silver chain, "
        "white sleeveless high-collar zip-up crop top, white cargo mini skirt, black belt with a long black strap, one black thigh strap on her right thigh only, "
        "silver bracelets, white ankle socks, white chunky sneakers. ")
STYLE = ("Clean Japanese anime cel shading with thin black line art, flat colors, anime background art. No text, no letters, no numbers, no logos. ")

SHOTS = {
    "s017": dict(
        refs=["refs/entities/hovercar/side.png", "refs/entities/drone/front.png"],
        qwen=("Image 1 is the character sheet, image 2 shows the hover car design (no wheels, glowing strip under the body, floats in the air), "
              "image 3 shows the small spherical white drone. Wide side view of a night shopping street, camera perpendicular to the street at eye level: "
              "the girl from image 1 runs toward the right of the image in the left half, full body, left foot forward, ponytail flying behind; "
              "the small drone from image 3 floats just behind her head, half hidden. In front of her on the right, three hover cars like image 2 hang motionless in midair "
              "in a rising row like stairs: the first car's roof is at her waist height, the second higher, the third higher still at second-floor height, all side views, "
              "level with the ground, no wheels. Behind them are shop fronts and raindrops frozen in midair. " + CHAR + STYLE),
        sdxl=("full body, from side, running, hover car, floating car, no wheels, row of cars in midair, shop fronts, night, street, frozen raindrops"),
        grey="all",   # 首幀全畫面灰階：她還沒踩到車（灰化由後製做）
    ),
    "s024": dict(
        refs=["refs/entities/street/frozen.png", "refs/entities/drone/front.png"],
        qwen=("Image 1 is the character sheet, image 2 shows the shopping street, image 3 shows the small spherical white drone. Low-angle view looking up about 40 degrees "
              "from the street: the girl from image 1 hangs frozen in midair high above the street, facing the camera, full body, her body leaning far forward, both arms "
              "stretched forward toward the camera with open hands, legs trailing behind, eyes open, determined; the small drone from image 3 floats at her left shoulder "
              "(right side of the image). Behind her are the upper floors of the shop buildings from image 2 with holographic billboards, raindrops hanging still in the air, "
              "dark night sky. " + CHAR + STYLE),
        sdxl=("full body, midair, from below, leaning forward, arms forward, reaching towards viewer, open hands, determined, shopping street, billboards, rain, night sky"),
        grey="all",   # 首幀全灰；[22,1] 的顏色爆發與衝擊波由後製做（見 C 的示意）
        wave=True,
    ),
}


def api(path, data=None):
    return L.api(path, data)


def run(wf, node):
    pid = api("/prompt", {"prompt": wf, "client_id": "p3kf"})["prompt_id"]
    while True:
        h = api("/history/" + pid)
        if pid in h and h[pid].get("outputs"):
            break
        if pid in h and h[pid]["status"].get("status_str") == "error":
            raise RuntimeError(json.dumps(h[pid]["status"])[:2000])
        time.sleep(1)
    im = h[pid]["outputs"][node]["images"][0]
    return urllib.request.urlopen(f"{HOST}/view?" + urllib.parse.urlencode(im)).read()


def put(path, name):
    COMFY_INPUT.mkdir(exist_ok=True)
    shutil.copy(path, COMFY_INPUT / name)
    return name


def qwen(prompt, refs, seed):
    wf = json.loads((ROOT / "workflows/p0_ref_gen.json").read_text())
    imgs = [put(ROOT / "refs/動畫_女生_LUNA.jpg", "p3kf_sheet.jpg")] + [put(ROOT / r, f"p3kf_ref{i}.png") for i, r in enumerate(refs, 2)]
    del wf["11"]   # 範本的 image2（CROP.png）；只給角色表時不能留著
    for k in ("20", "21"):
        wf[k]["inputs"].pop("image2", None)
    for i, name in enumerate(imgs, 1):
        wf[str(9 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        for k in ("20", "21"):
            wf[k]["inputs"][f"image{i}"] = [str(9 + i), 0]
    wf["20"]["inputs"]["prompt"] = prompt
    wf["21"]["inputs"]["prompt"] = ""
    wf["6"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}}
    wf["4"]["inputs"]["model"] = ["6", 0]
    wf["30"]["inputs"].update(width=W, height=H)
    wf["40"]["inputs"].update(seed=seed, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple")
    wf["60"]["inputs"]["filename_prefix"] = "p3kf/a"
    return run(wf, "60")


def sdxl_img2img(src_name, tags, seed, denoise, canny):
    pos = L.BASE + L.prompt_outfit(tags) + tags + L.QUALITY
    wf = L.workflow(LORA, 0.9, pos, W, H, seed, 28, 6.0)
    wf["30"] = {"class_type": "LoadImage", "inputs": {"image": src_name}}
    wf["31"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["30", 0], "vae": ["1", 2]}}
    wf["7"]["inputs"]["latent_image"] = ["31", 0]
    wf["7"]["inputs"]["denoise"] = denoise
    del wf["6"]
    if canny:
        wf["20"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": L.CONTROLNET}}
        wf["21"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["20", 0], "type": "canny/lineart/anime_lineart/mlsd"}}
        wf["22"] = {"class_type": "Canny", "inputs": {"image": ["30", 0], "low_threshold": 0.3, "high_threshold": 0.6}}
        wf["23"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "control_net": ["21", 0], "image": ["22", 0],
                                                                         "strength": 0.6, "start_percent": 0.0, "end_percent": 0.6, "vae": ["1", 2]}}
        wf["7"]["inputs"]["positive"], wf["7"]["inputs"]["negative"] = ["23", 0], ["23", 1]
    return run(wf, "9"), pos


def wave_demo(img, center, radius):
    """@broadcast_wave 示意：白芯薄光環＋青、洋紅光暈；環內恢復顏色、環外灰階（實際由後製逐幀做）。"""
    grey = ImageOps.grayscale(img).convert("RGB")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)
    out = Image.composite(img, grey, mask.filter(ImageFilter.GaussianBlur(6)))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    for dr, col in [(10, (208, 64, 200, 150)), (-10, (41, 211, 245, 150)), (0, (244, 248, 255, 255))]:
        r = radius + dr
        d.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline=col, width=6 if dr == 0 else 10)
    glow = glow.filter(ImageFilter.GaussianBlur(3))
    return Image.alpha_composite(out.convert("RGBA"), glow).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    for shot, cfg in SHOTS.items():
        if a.only and shot not in a.only.split(","):
            continue
        out = ROOT / "refs/keyframe_test" / shot
        out.mkdir(parents=True, exist_ok=True)
        log = out / "results.json"
        recs = json.loads(log.read_text()) if log.exists() else []
        for k in range(a.seeds):
            seed = 20261124 + k
            fa = out / f"A_qwen_s{k}.png"
            if not fa.exists():
                t0 = time.time()
                fa.write_bytes(qwen(cfg["qwen"], cfg["refs"], seed))
                recs.append(dict(step="A", file=str(fa.relative_to(ROOT)), seed=seed, refs=cfg["refs"], prompt=cfg["qwen"], seconds=round(time.time() - t0, 1)))
            src = put(fa, f"p3kf_{shot}_A{k}.png")
            for tag, denoise, canny in [("B1_img2img", 0.45, False), ("B2_canny", 0.6, True)]:
                fb = out / f"{tag}_s{k}.png"
                if fb.exists():
                    continue
                t0 = time.time()
                data, pos = sdxl_img2img(src, cfg["sdxl"], seed, denoise, canny)
                fb.write_bytes(data)
                recs.append(dict(step=tag, file=str(fb.relative_to(ROOT)), seed=seed, denoise=denoise, canny=canny, prompt=pos, seconds=round(time.time() - t0, 1)))
                img = Image.open(fb).convert("RGB")
                if cfg.get("grey") == "all":
                    ImageOps.grayscale(img).convert("RGB").save(out / f"C_{tag}_grey_s{k}.png")
                if cfg.get("wave"):
                    wave_demo(img, (W // 2, int(H * 0.45)), int(H * 0.38)).save(out / f"C_{tag}_wave_s{k}.png")
            log.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
            print(shot, k, flush=True)


if __name__ == "__main__":
    main()
