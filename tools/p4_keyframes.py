#!/usr/bin/env python3
"""P4 試作 s002/s003/s004 關鍵幀（G0 定案管線：Qwen 合成 A → Illustrious+LoRA canny B2 → 後製 C）。

~/creative/ComfyUI/comfyui/bin/python tools/p4_keyframes.py s003 [--seeds 1,2,3] [--stage A|B|all]
候選存 shots/<id>/candidates/，prompt/seed 記在 candidates/results.json。選定的圖手動複製成 keyframe.png/last.png。
"""
import argparse, json, shutil, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_keyframe_test as K

ROOT = K.ROOT
W, H = K.W, K.H
CHAR = K.CHAR
NOTXT = "No text, no letters, no numbers, no logos anywhere. "

SDXL3 = ("upper body, standing, closed eyes, facing viewer, calm, in glass tube, stasis pod, glass, white light from below, dark room, blurry background, "
        "screens in background, greyscale background, cold blue-black")
SDXL4 = ("extreme close-up, close-up, eyes focus, closed eyes, long eyelashes, eyebrows, bangs, looking at viewer, light from below, "
        "dark background, glass reflection")
# 每個 job：name, 鏡號, qwen prompt, qwen refs, sdxl tags, 是否用 LoRA B2（False 則只留 A）, 其他
JOBS = {
 "s002": [dict(name="kf", refs=["refs/entities/hq_control/wide.png"],
   qwen=("Image 1 is the character sheet, image 2 shows the circular control room layout with a curved wall of monitors; ignore the capsule in the middle of image 2. "
         "Eye-level wide shot of the empty circular underground control room, perfectly symmetrical composition, looking at the far curved wall. "
         "The curved back wall spans the full width of the image and holds only 21 large rectangular screens: each row has just seven big screens side by side, "
         "so seven columns and three rows, with the outer columns cut off by the image edges, thick dark blue frames and wide gaps between the screens, "
         "the grid occupying the vertical range from 12 percent to 78 percent of the image height. "
         "Twenty of the screens are completely switched off, plain black glass. Only the one screen in the centre column and middle row is lit: "
         "it is a black screen showing a thin cyan zigzag waveform line that stays inside that screen. "
         "There is nothing standing in the middle of the room, the centre of the floor is empty. "
         "The black metal grid floor has thin cyan guide light strips running from the bottom edge of the image converging toward the empty centre of the floor. "
         "Very dark blue-black room, flat anime background art. " + K.STYLE),
   sdxl="no humans, scenery, indoors, dark room, control room, curved wall, monitors, many black screens, glass capsule, cyan light strips on floor, symmetrical, dark",
   lora=False, name_="room")],
 "s003": [
  dict(name="kf", refs=["refs/entities/pod/closed.png", "refs/entities/monitor_wall/frozen.png"],
   qwen=("Image 1 is the character sheet, image 2 shows the glass capsule pod, image 3 shows a wall of grey screens. "
         "Close medium shot cropped at the hips, eye level, straight-on front view: the girl from image 1 stands inside the transparent glass tube of the capsule from image 2, "
         "seen only from the waist up, the bottom edge of the frame cuts her at the hips and her thighs are not visible, her eyes closed, calm peaceful expression, arms relaxed at her sides, her head close to the top of the frame. "
         "The bright white top surface of the capsule base below the frame glows and softly lights her face from underneath. "
         "Behind the capsule is a wall of greyscale monitor screens, completely out of focus and blurred, dark room. " + CHAR + K.STYLE),
   sdxl=SDXL3),
  dict(name="last", refs=["{kf}", "refs/entities/drone/front.png"], src="kf",
   qwen=("Image 1 is the character sheet, image 2 is the first frame, image 3 shows a small spherical white drone with a black visor band. "
         "Keep image 2 exactly as it is, the same girl, glass capsule, pose, framing and blurred grey screens in the background, and add one small floating drone "
         "from image 3 in the upper right area of the frame, in front of the capsule, with its centre at 78 percent of the image width and 22 percent of the image height. "
         "The drone is small, its diameter is only one quarter of the image height, much smaller than the girl's head and shoulders width, "
         "and the visor band faces the camera, covering only the front third of the sphere. " + K.STYLE),
   sdxl=SDXL3 + ", floating white sphere drone with black visor band, drone in upper right"),
 ],
 "s004": [
  dict(name="kf", refs=["refs/04_expr_happy.png"], zoom=1.45, zc=(0.5, 0.47),
   qwen=("Image 1 is the character sheet, image 2 shows the same girl's face as a reference for the face and eye style. "
         "Extreme close-up in a wide 16:9 frame, straight-on: only the girl's two eyes, eyebrows and the bridge of the nose fill the entire frame, "
         "cropped from the forehead bangs at the top to just below the eyes at the bottom. Both eyes are gently closed, long clear black eyelashes along the closed lids, calm peaceful face. "
         "The skin is lit by soft white light from below, and the top part of the image near the dark purple bangs is slightly darker. "
         "Seen through a pane of transparent glass with a very faint light reflection. Dark background. " + K.STYLE),
   sdxl=SDXL4),
  dict(name="last", refs=["{kf}"], src="kf",
   qwen=("Image 1 is the character sheet, image 2 is the first frame, an extreme close-up of the girl's closed eyes. "
         "Keep image 2 exactly as it is, with the same framing, lighting, hair, eyebrows, skin and glass, and only change the eyes: both eyes are now wide open, "
         "looking straight at the camera, large round irises in dark purple-grey with a small white highlight, the pupils clearly visible, the long eyelashes on the upper lids. " + K.STYLE),
   sdxl=SDXL4.replace("closed eyes", "open eyes, looking at viewer, grey eyes, purple-grey eyes")),
 ],
}


def log(shot, rec):
    p = ROOT / "shots" / shot / "candidates" / "results.json"
    r = json.loads(p.read_text()) if p.exists() else []
    r.append(rec)
    p.write_text(json.dumps(r, ensure_ascii=False, indent=1))


def run_job(shot, job, seed, stage="all", denoise=0.6):
    d = ROOT / "shots" / shot / "candidates"
    d.mkdir(parents=True, exist_ok=True)
    n = job["name"]
    fa = d / f"{n}_A_s{seed}.png"
    refs = [r.replace("{kf}", f"shots/{shot}/candidates/kf_B2_s{job.get('srcseed', seed)}.png") for r in job["refs"]]
    if not fa.exists():
        fa.write_bytes(K.qwen(job["qwen"], refs, seed))
        log(shot, dict(step="A", file=fa.name, seed=seed, refs=job["refs"], prompt=job["qwen"]))
    if stage == "A" or job.get("lora") is False:
        return fa
    fb = d / f"{n}_B2_s{seed}.png"
    if not fb.exists():
        srcf = fa
        if job.get("zoom"):   # 先把 A 以眼睛為中心放大裁切，再 B2（Qwen 的 ECU 常常拍太寬）
            z, (zx, zy) = job["zoom"], job.get("zc", (0.5, 0.5))
            im = Image.open(fa).convert("RGB")
            cw, ch = W / z, H / z
            x0 = min(max(zx * W - cw / 2, 0), W - cw)
            y0 = min(max(zy * H - ch / 2, 0), H - ch)
            srcf = d / f"{n}_Az_s{seed}.png"
            im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((W, H), Image.LANCZOS).save(srcf)
        src = K.put(srcf, f"p4_{shot}_{n}_{seed}.png")
        data, pos = K.sdxl_img2img(src, job["sdxl"], seed, denoise, True)
        fb.write_bytes(data)
        log(shot, dict(step="B2", file=fb.name, seed=seed, denoise=denoise, canny=True, prompt=pos))
    return fb


def contact(shot, files, out):
    ims = [Image.open(f).convert("RGB") for f in files]
    tw = 672
    th = int(tw * 9 / 16) if ims[0].width > ims[0].height else 400
    cols = 2 if len(ims) > 2 else len(ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 20)), "black")
    dr = ImageDraw.Draw(sheet)
    for i, (im, f) in enumerate(zip(ims, files)):
        im = ImageOps.contain(im, (tw, th))
        x, y = (i % cols) * tw, (i // cols) * (th + 20)
        sheet.paste(im, (x, y + 20))
        dr.text((x + 4, y + 4), Path(f).name, fill="white")
    sheet.save(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--job", default="")
    ap.add_argument("--srcseed", type=int, default=0, help="last 以哪個 kf 候選的 seed 為底")
    ap.add_argument("--denoise", type=float, default=0.6)
    a = ap.parse_args()
    for job in JOBS[a.shot]:
        if a.job and job["name"] != a.job:
            continue
        if a.srcseed:
            job["srcseed"] = a.srcseed
        for s in a.seeds.split(","):
            print(a.shot, job["name"], s, run_job(a.shot, job, int(s), a.stage, a.denoise), flush=True)


def cutout_drone(ref="refs/entities/drone/front.png"):
    """從參考圖摳出 ECHO：由四邊洪水填充去掉白底與地面陰影，剩下的（含外殼白色部分）就是球體。"""
    import numpy as np
    from scipy import ndimage as ndi
    im = Image.open(ROOT / ref).convert("RGB")
    a = np.asarray(im).astype(int)
    bgish = (a.min(axis=2) > 225) | ((a.max(axis=2) - a.min(axis=2) < 22) & (a.min(axis=2) > 170))   # 白底與淺灰陰影
    lab, n = ndi.label(bgish)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    fg = ~bg
    fg = ndi.binary_fill_holes(fg)
    fg = ndi.binary_opening(fg, iterations=3)
    lab, n = ndi.label(fg)
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    fg = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(fg)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    m = Image.fromarray((fg * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.2))
    rgba = im.copy().convert("RGBA")
    rgba.putalpha(m)
    return rgba.crop(box)


def paste_drone(base_path, out_path, cx=0.78, cy=0.22, diam=0.25):
    base = Image.open(base_path).convert("RGBA")
    d = cutout_drone()
    s = diam * base.height / max(d.size)
    d = d.resize((round(d.width * s), round(d.height * s)), Image.LANCZOS)
    base.alpha_composite(d, (round(cx * base.width - d.width / 2), round(cy * base.height - d.height / 2)))
    base.convert("RGB").save(out_path)


def eye_composite(kf, openimg, out, scale=1.07, eyes=((326, 440), (996, 440)), box=(290, 230)):
    """把 openimg（睜眼版）的雙眼區域縮放對位後，羽化貼到 kf 上，其他部分保持 kf 不變。"""
    k = Image.open(kf).convert("RGB")
    o = Image.open(openimg).convert("RGB")
    cx, cy = W / 2, 440
    o = o.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    ox, oy = round(cx - cx * scale), round(cy - cy * scale)
    big = Image.new("RGB", (W, H))
    big.paste(o, (ox, oy))
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    for ex, ey in eyes:
        d.ellipse((ex - box[0] / 2 - 40, ey - box[1] / 2 - 40, ex + box[0] / 2 + 40, ey + box[1] / 2 + 40), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(24))
    Image.composite(big, k, mask).save(out)
    mask.save(Path(out).with_suffix(".mask.png"))


def sdxl_inpaint(src_name, mask_name, tags, seed, denoise=0.5):
    """B2 之後的局部重繪：只在遮罩內以 LoRA 重畫（用於 s004 把睜眼區域與 keyframe 融合）。"""
    L = K.L
    pos = L.BASE + L.prompt_outfit(tags) + tags + L.QUALITY
    wf = L.workflow(K.LORA, 0.9, pos, W, H, seed, 28, 6.0)
    wf["30"] = {"class_type": "LoadImage", "inputs": {"image": src_name}}
    wf["31"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["30", 0], "vae": ["1", 2]}}
    wf["32"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
    wf["33"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["31", 0], "mask": ["32", 0]}}
    wf["7"]["inputs"]["latent_image"] = ["33", 0]
    wf["7"]["inputs"]["denoise"] = denoise
    del wf["6"]
    return K.run(wf, "9"), pos


def compose_s002(room, podsrc, out, pod_h=0.25, base_y=0.87, cx=0.519, dim=0.72, zoom=1.4, zy=110):
    """空房間（Qwen）＋從另一張含艙畫面切出的啟動艙，縮到畫面高 22%、底座落在地面圓心；艙體稍微調暗（底座白光很暗）。"""
    room_im = Image.open(room).convert("RGB")
    if zoom > 1:   # Qwen 的牆會畫成約 11 欄；放大裁切成約 7 欄（水平置中，垂直讓主螢幕落在畫面高約 45%）
        cw, ch = W / zoom, H / zoom
        x0 = (W - cw) / 2
        room_im = room_im.crop((round(x0), zy, round(x0 + cw), round(zy + ch))).resize((W, H), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 60, 2))
    room_im = room_im.convert("RGBA")
    if podsrc is None:
        room_im.convert("RGB").save(out)
        return
    src = Image.open(podsrc).convert("RGB")
    cx0, cy0, cx1, cy1 = 540, 180, 800, 640           # 艙在來源圖的裁切框（1344×768 座標）
    crop = src.crop((cx0, cy0, cx1, cy1)).resize(((cx1 - cx0) * 2, (cy1 - cy0) * 2), Image.LANCZOS)
    m = Image.new("L", crop.size, 0)
    d = ImageDraw.Draw(m)
    d.ellipse((140, 95, 465, 310), fill=255)           # 圓頂
    d.rectangle((140, 200, 465, 790), fill=255)        # 玻璃與底座
    d.ellipse((132, 745, 470, 832), fill=255)          # 底座下緣
    m = m.filter(ImageFilter.GaussianBlur(1.5))
    crop = Image.eval(crop, lambda v: int(v * dim))
    pod = crop.convert("RGBA")
    pod.putalpha(m)
    bbox = m.point(lambda v: 255 if v > 40 else 0).getbbox()
    pod = pod.crop(bbox)
    s = pod_h * room_im.height / pod.height
    pod = pod.resize((round(pod.width * s), round(pod.height * s)), Image.LANCZOS)
    room_im.alpha_composite(pod, (round(cx * room_im.width - pod.width / 2), round(base_y * room_im.height - pod.height)))   # (cx, base_y)＝放大後地面內圈圓心（約 698, 665）
    room_im.convert("RGB").save(out)


def screens_s002(room_nopod, out_mask, out_json):
    """在無艙的放大房間圖上找出 7×3 螢幕（黑色連通區域取 bbox，邊緣被切掉的欄保留可見部分）；
    被波形切開（主螢幕）或缺漏的格子用左右鄰格內插。輸出 screens_mask.png 與 screens.json（col 0..6 由左到右、row 0..2 由上到下）。"""
    import numpy as np
    from scipy import ndimage as ndi
    a = np.asarray(Image.open(room_nopod).convert("RGB")).astype(int)
    dark = ndi.binary_closing(a.max(axis=2) < 24, iterations=1)
    lab, n = ndi.label(dark)
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if w > 40 and 55 < h < 150 and (lab[sl] == i).sum() > 0.6 * w * h and 120 < sl[0].start + h / 2 < 520:
            comps.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
    ycs = sorted((c[1] + c[3]) / 2 for c in comps)
    rowc = [190, 330, 460]   # 三層大致的中心高度（由偵測結果得到）
    grid = {}
    for c in comps:
        r = min(range(3), key=lambda k: abs((c[1] + c[3]) / 2 - rowc[k]))
        col = round(((c[0] + c[2]) / 2 - W / 2) / 190) + 3
        if 0 <= col <= 6 and (r, col) not in grid:
            grid[(r, col)] = c
    interp = []
    for r in range(3):
        for col in range(7):
            if (r, col) in grid:
                continue
            nb = [grid[(r, k)] for k in (col - 1, col + 1) if (r, k) in grid]
            if len(nb) == 2:
                grid[(r, col)] = [round((nb[0][i] + nb[1][i]) / 2) for i in range(4)]
            elif nb:
                grid[(r, col)] = nb[0]
            interp.append([r, col])
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for b in grid.values():
        d.rectangle(b, fill=255)
    m.save(out_mask)
    json.dump(dict(size=[W, H], note="bbox=[x0,y0,x1,y1]，key 為 'row,col'；row 0 上層、col 0 最左；主螢幕是 '1,3'；interpolated 是由鄰格內插的格子（被波形切開或缺漏）",
                   main="1,3", interpolated=interp, screens={f"{r},{c}": b for (r, c), b in sorted(grid.items())}),
              open(out_json, "w"), indent=1)
    return grid
