#!/usr/bin/env python3
"""P6 關鍵幀候選（依段落、可重用）。方法與 P4 相同（G0 定案）：
  A  Qwen-Image-Edit 合成構圖（角色表＋plan/p6_jobs.yaml 指定的 refs，英文說明寫在 jobs 檔）
  B  Illustrious＋LUNA LoRA v2 img2img（denoise 約 0.6）＋canny ControlNet；路線 B 的鏡頭再加 OpenPose 骨架（p3_lora_gen.add_controlnet）
  C  後製：灰化（frozen）、貼 ECHO（p4_keyframes.paste_drone）、程式繪製（s001 波形平線、s007 遠處平台）等
  最後一步固定把 1344×768 的中間稿置中裁成 1280×720，輸出 candidates/<frame>_s<seed>.png

~/creative/ComfyUI/comfyui/bin/python tools/p6_keyframes.py --shots s001,s005            # 全部幀、jobs 檔預設的種子
~/creative/ComfyUI/comfyui/bin/python tools/p6_keyframes.py --section hook --frames first  # 一個段落的首幀
~/creative/ComfyUI/comfyui/bin/python tools/p6_keyframes.py --shots s006 --frames first --seeds 3 --stage post --force   # 只重跑某幀某 seed 的後製
~/creative/ComfyUI/comfyui/bin/python tools/p6_keyframes.py --shots s006 --contact                                         # 只重拼對照圖
~/creative/ComfyUI/comfyui/bin/python tools/p6_keyframes.py --shots s006 --reject first:3:手指多一根           # 剔除候選（對照圖不放）
~/creative/ComfyUI/comfyui/bin/python tools/p6_keyframes.py --shots s006 --record                                          # 寫 keyframe.json

檔名（shots/<id>/candidates/，已在 .gitignore）：
  <frame>_A_s<seed>.png  Qwen 中間稿        <frame>_B_s<seed>.png  LoRA 中間稿（1344×768）
  <frame>_s<seed>.png    最終候選（1280×720，對照圖只放這種）
prompt、seed、refs、denoise、耗時記在 candidates/results.json；剔除記在 candidates/rejects.json。
不建立 keyframe.png／last.png／mid.png，那由使用者挑。
"""
import argparse, json, math, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_keyframe_test as K
import p3_lora_gen as L
import p4_keyframes as P4

ROOT = K.ROOT
W, H = K.W, K.H            # 中間稿 1344×768（SDXL bucket，同 P4）
OUT_W, OUT_H = 1280, 720   # 最終候選
JOBS_FILE = ROOT / "plan/p6_jobs.yaml"
FRAMES = ("first", "mid", "last")
FONT = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"

# base 提示裡的固定外觀 tag：p5_prompts 的 tag 把它們寫在前面，p3_lora_gen.prompt_outfit 會依構圖重寫，所以先剝掉
_FIXED = {t.strip() for t in (
    "lunacsf, 1girl, solo, grey eyes, dark purple hair, purple hair, high ponytail, braided hair, hair chain, earrings, high collar, "
    "white shirt, sleeveless, white crop top, zipper, midriff, bracelets, white skirt, cargo skirt, black belt, single thigh strap, "
    "white socks, white sneakers, masterpiece, best quality, anime screencap, clean lineart, cel shading").split(",")}


# ---------------------------------------------------------------- 讀檔
def load_jobs():
    j = yaml.safe_load(JOBS_FILE.read_text())
    return j


def load_shots():
    return {s["id"]: s for s in yaml.safe_load((ROOT / "plan/shots.yaml").read_text())["shots"]}


def p5_tags(shot, frame):
    p = yaml.safe_load((ROOT / "plan/p5_prompts.yaml").read_text())["shots"][shot]["keyframe"]
    t = p.get({"first": "first", "last": "last"}.get(frame, "first"))
    return t


def strip_tags(tags):
    """去掉 p5 tag 裡的固定外觀與品質 tag，只留構圖／場景 tag（之後由 L.BASE＋prompt_outfit＋L.QUALITY 重組）。"""
    return ", ".join(t.strip() for t in tags.split(",") if t.strip() and t.strip() not in _FIXED)


def expand(text, macros):
    for k, v in macros.items():
        text = text.replace(f"<<{k}>>", v.strip() + " ")
    return " ".join(text.split()) + " "


# ---------------------------------------------------------------- 路徑與記錄
def cdir(shot):
    d = ROOT / "shots" / shot / "candidates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fA(shot, frame, seed): return cdir(shot) / f"{frame}_A_s{seed}.png"
def fB(shot, frame, seed): return cdir(shot) / f"{frame}_B_s{seed}.png"
def fF(shot, frame, seed): return cdir(shot) / f"{frame}_s{seed}.png"


def log(shot, rec):
    p = cdir(shot) / "results.json"
    r = json.loads(p.read_text()) if p.exists() else []
    r.append(rec)
    p.write_text(json.dumps(r, ensure_ascii=False, indent=1))


def rejects(shot):
    p = cdir(shot) / "rejects.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- OpenPose 骨架
_LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
_COL = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170],
        [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
_KP = ["nose", "neck", "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri", "r_hip", "r_kne", "r_ank", "l_hip", "l_kne", "l_ank", "r_eye", "l_eye", "r_ear", "l_ear"]


def draw_pose(kp, box, size, out, canvas=None):
    """kp：{部位: [x, y]}（r_/l_ 是『角色自己的』右左；x、y 為整幅 1344×768 的比例）；box＝這張骨架圖對應的整幅裁切框（像素）；
    size＝輸出像素。缺的點略過（背面不給臉部點）。標準 OpenPose 色與肢節順序。"""
    x0, y0, x1, y1 = box
    cw, ch = canvas or (W, H)
    sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)
    pts = {i: ((kp[n][0] * cw - x0) * sx, (kp[n][1] * ch - y0) * sy) for i, n in enumerate(_KP) if n in kp}
    im = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(im)
    sw = max(3, round(4 * size[1] / 512))
    for i, (a, b) in enumerate(_LIMBS):
        if a in pts and b in pts:
            d.line([pts[a], pts[b]], fill=tuple(_COL[i]), width=sw)
    for i, p in pts.items():
        d.ellipse((p[0] - sw, p[1] - sw, p[0] + sw, p[1] + sw), fill=tuple(_COL[i]))
    im.save(out)
    return out


def edge_map(src, out, lo=80, hi=160):
    """A 圖的 canny 邊緣（白線黑底）當 add_controlnet 的參考線稿。"""
    import cv2
    g = cv2.cvtColor(np.asarray(Image.open(src).convert("RGB")), cv2.COLOR_RGB2GRAY)
    Image.fromarray(cv2.Canny(g, lo, hi)).convert("RGB").save(out)
    return out


# ---------------------------------------------------------------- 步驟 A：Qwen
def step_A(shot, frame, seed, v, macros, tmp_refs):
    fa = fA(shot, frame, seed)
    refs = []
    for r in v.get("refs", []):
        m = r.replace("{self}", f"shots/{shot}/candidates/{frame}_B_s{seed}.png") if "{self}" in r else r
        for fr in FRAMES:   # {first}／{last}／{mid}＝同鏡該幀的 LoRA／A 中間稿（有 B 用 B，沒有就用 A）；{first:final}＝最終稿
            for suffix, kind in (("{%s:final}" % fr, "final"), ("{%s}" % fr, "mid")):
                if suffix in m:
                    sd = v.get("srcseed", seed)
                    cand = fF(shot, fr, sd) if kind == "final" else (fB(shot, fr, sd) if fB(shot, fr, sd).exists() else fA(shot, fr, sd))
                    m = m.replace(suffix, str(cand.relative_to(ROOT)))
        refs.append(m)
    prompt = expand(v["qwen"], macros)
    t0 = time.time()
    fa.write_bytes(K.qwen(prompt, refs, seed))
    log(shot, dict(step="A", frame=frame, file=fa.name, seed=seed, refs=refs, prompt=prompt, seconds=round(time.time() - t0, 1)))
    return fa


def apply_ops(im, ops, ctx):
    done = []
    for st in ops or []:
        st = dict(st)
        op = st.pop("op")
        if op == "black_line":
            im = op_black_line(im, st, ctx[2])
        elif op == "shrink_paste":
            im = op_shrink_paste(im, st, ctx)
        elif op == "skyline_from":
            im = op_skyline_from(im, st, ctx)
        elif op == "roof":
            im = op_roof(im, st, ctx[2])
        elif op == "paste_sprite":
            im = op_paste_sprite(im, st, ctx)
        else:
            im = OPS[op](im, st)
        done.append(dict(op=op, **st))
    return im, done


# ---------------------------------------------------------------- 步驟 B：Illustrious＋LoRA（canny；路線 B 加 OpenPose）
def step_B(shot, frame, seed, v, src):
    fb = fB(shot, frame, seed)
    d = cdir(shot)
    cn = v.get("controlnet", {}) or {}
    denoise = v.get("denoise", 0.6)
    tags = v.get("tags") or strip_tags(p5_tags(shot, frame))
    if v.get("tags_add"):
        tags += ", " + v["tags_add"]
    for t in v.get("tags_drop", []):
        tags = ", ".join(x for x in (y.strip() for y in tags.split(",")) if x != t)
    pos = L.BASE + L.prompt_outfit(tags) + tags + L.QUALITY
    im = Image.open(src).convert("RGB")
    im, _ = apply_ops(im, v.get("pre"), (shot, frame, seed))   # B 之前先對 A 做的構圖修正（放大裁切、平移、縮貼）
    box = v.get("region")
    box = tuple(box) if box else (0, 0, W, H)
    pose_img = None
    if box != (0, 0, W, H):
        cw, ch = box[2] - box[0], box[3] - box[1]
        s = v.get("region_scale") or min(1216 / max(cw, ch), 4.0)
        sw, sh = round(cw * s / 8) * 8, round(ch * s / 8) * 8
        crop = im.crop(box).resize((sw, sh), Image.LANCZOS)
    else:
        crop, (sw, sh) = im, (W, H)
    cp = d / f"{frame}_Bin_s{seed}.png"
    crop.save(cp)
    inp = K.put(cp, f"p6_{shot}_{frame}_{seed}.png")
    wf = L.workflow(K.LORA, 0.9, pos, sw, sh, seed, 28, 6.0)
    wf["30"] = {"class_type": "LoadImage", "inputs": {"image": inp}}
    wf["31"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["30", 0], "vae": ["1", 2]}}
    wf["7"]["inputs"]["latent_image"] = ["31", 0]
    wf["7"]["inputs"]["denoise"] = denoise
    del wf["6"]
    cnrec = {}
    if cn.get("pose"):
        pose = cn["pose"]
        if isinstance(pose, dict):   # 內嵌關鍵點
            pp = d / f"{frame}_pose_s{seed}.png"
            draw_pose(pose, box, (sw, sh), pp)
            pose_rel = str(pp.relative_to(ROOT))
        else:
            pose_rel = pose
        lp = edge_map(cp, d / f"{frame}_lineart_s{seed}.png") if cn.get("canny", True) else None
        L.add_controlnet(wf, pose_rel, str(lp.relative_to(ROOT)) if lp else None, cn.get("canny_strength", 0.6))
        # img2img 只走後 denoise 比例的步數，add_controlnet 預設的 0.35 結束點對 img2img 等於沒作用，所以依 jobs 覆寫
        applies = [k for k in wf if wf[k].get("class_type") == "ControlNetApplyAdvanced"]
        pose_node = [k for k in applies if wf[wf[k]["inputs"]["control_net"][0]]["inputs"]["type"] == "openpose"]
        for k in applies:
            if k in pose_node:
                wf[k]["inputs"].update(strength=cn.get("pose_strength", 1.0), end_percent=cn.get("pose_end", 0.9))
            else:
                wf[k]["inputs"].update(strength=cn.get("canny_strength", 0.6), end_percent=cn.get("canny_end", 0.7))
        cnrec = dict(pose=pose_rel, pose_strength=cn.get("pose_strength", 1.0), pose_end=cn.get("pose_end", 0.9),
                     canny=bool(lp), canny_strength=cn.get("canny_strength", 0.6), canny_end=cn.get("canny_end", 0.7))
    elif cn.get("canny", True):
        wf["20"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": L.CONTROLNET}}
        wf["21"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["20", 0], "type": "canny/lineart/anime_lineart/mlsd"}}
        wf["22"] = {"class_type": "Canny", "inputs": {"image": ["30", 0], "low_threshold": 0.3, "high_threshold": 0.6}}
        wf["23"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "control_net": ["21", 0], "image": ["22", 0],
                                                                         "strength": cn.get("canny_strength", 0.6), "start_percent": 0.0,
                                                                         "end_percent": cn.get("canny_end", 0.6), "vae": ["1", 2]}}
        wf["7"]["inputs"]["positive"], wf["7"]["inputs"]["negative"] = ["23", 0], ["23", 1]
        cnrec = dict(canny=True, canny_strength=cn.get("canny_strength", 0.6), canny_end=cn.get("canny_end", 0.6))
    t0 = time.time()
    data = K.run(wf, "9")
    out = d / f"{frame}_Bout_s{seed}.png"
    out.write_bytes(data)
    res = Image.open(out).convert("RGB")
    if box != (0, 0, W, H):   # 區域重繪：縮回原尺寸，羽化貼回
        res = res.resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS)
        full = im.copy()
        feather = v.get("feather", 28)
        m = Image.new("L", im.size, 0)
        ImageDraw.Draw(m).rectangle((box[0] + feather, box[1] + feather, box[2] - feather, box[3] - feather), fill=255)
        for edge, hit in (("l", box[0] <= 0), ("r", box[2] >= W), ("t", box[1] <= 0), ("b", box[3] >= H)):   # 貼齊畫面邊的那側不羽化
            if hit:
                ImageDraw.Draw(m).rectangle({"l": (0, box[1], box[0] + feather, box[3]), "r": (box[2] - feather, box[1], W, box[3]),
                                             "t": (box[0], 0, box[2], box[1] + feather), "b": (box[0], box[3] - feather, box[2], H)}[edge], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(feather / 2.5))
        layer = Image.new("RGB", im.size)
        layer.paste(res, box[:2])
        res = Image.composite(layer, full, m)
    res.save(fb)
    out.unlink()
    log(shot, dict(step="B", frame=frame, file=fb.name, seed=seed, denoise=denoise, pre=v.get("pre"), controlnet=cnrec, region=list(box) if box != (0, 0, W, H) else None,
                   prompt=pos, seconds=round(time.time() - t0, 1)))
    return fb


# ---------------------------------------------------------------- 步驟 C：後製
def op_grey(im, p):
    """灰化（style bible：飽和度 0、亮度不變）。where 依序畫上遮罩：{rect:[x0,y0,x1,y1] 比例, grey: true/false, feather: px}；
    grey=true 的區域去飽和，false 的區域保持彩色（LUNA、ECHO、彩色街區）。ellipse 同理。"""
    m = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(m)
    for w in p["where"]:
        shape = w.get("rect") or w.get("ellipse")
        box = [shape[0] * im.width, shape[1] * im.height, shape[2] * im.width, shape[3] * im.height]
        tmp = Image.new("L", im.size, 0)
        td = ImageDraw.Draw(tmp)
        (td.rectangle if w.get("rect") else td.ellipse)(box, fill=255)
        if w.get("feather", 0):
            tmp = tmp.filter(ImageFilter.GaussianBlur(w["feather"]))
        if w.get("grey", True):
            m = Image.fromarray(np.maximum(np.asarray(m), np.asarray(tmp)))
        else:
            m = Image.fromarray((np.asarray(m).astype(float) * (1 - np.asarray(tmp) / 255.0)).astype("uint8"))
    grey = ImageOps.grayscale(im).convert("RGB")
    return Image.composite(grey, im, m)


def op_paste_drone(im, p):
    tmp = ROOT / "shots" / ".p6_tmp_drone.png"
    im.save(tmp)
    P4.paste_drone(tmp, tmp, p["cx"], p["cy"], p["diam"])
    out = Image.open(tmp).convert("RGB")
    tmp.unlink()
    return out


def op_black_line(im, p, seed):
    """s001：黑色螢幕玻璃面＋一條水平青線（程式繪製，位置與長度精確）。seed 只用來微調玻璃反光與光暈（候選之間有小差異）。"""
    rnd = np.random.RandomState(seed)
    a = np.zeros((H, W, 3), float) + np.array([4, 5, 7])
    yy, xx = np.mgrid[0:H, 0:W]
    ang = rnd.uniform(0.35, 0.6)
    refl = np.clip(np.sin((xx * math.cos(ang) + yy * math.sin(ang)) / rnd.uniform(220, 380) + rnd.uniform(0, 6)), 0, 1) ** 6
    a += refl[..., None] * p.get("refl", rnd.uniform(3, 7)) * np.array([0.6, 0.8, 1.0])
    vig = 1 - 0.25 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    a *= vig[..., None]
    base = Image.fromarray(a.clip(0, 255).astype("uint8"))
    y = p.get("y", 0.45) * H
    x0, x1 = p.get("x0", 0.1) * W, p.get("x1", 0.9) * W
    lw = max(3, round(0.005 * H * p.get("lw_scale", 1.0)))     # glossary：線寬約為螢幕高的 0.5%
    glow = Image.new("RGB", (W, H), 0)
    gd = ImageDraw.Draw(glow)
    gd.line([(x0, y), (x1, y)], fill=(41, 211, 245), width=lw * 3)
    glow = glow.filter(ImageFilter.GaussianBlur(lw * rnd.uniform(2.2, 3.4)))   # 光暈約 3 倍線寬
    core = Image.new("RGB", (W, H), 0)
    ImageDraw.Draw(core).line([(x0, y), (x1, y)], fill=(41, 211, 245), width=lw)
    core = core.filter(ImageFilter.GaussianBlur(0.8))
    out = np.asarray(base).astype(float) + np.asarray(glow).astype(float) * p.get("glow", 0.55) + np.asarray(core).astype(float)
    return Image.fromarray(out.clip(0, 255).astype("uint8"))


def op_platform_dot(im, p):
    """s007 首幀：豎井正中很深處的發光小圓盤（直徑 diam×畫面高）＋白衣紫髮小人影＋圓盤右緣的 ECHO 小白點。4 倍超取樣。"""
    S = 4
    big = im.resize((W * S, H * S), Image.LANCZOS).convert("RGBA")
    cx, cy = p.get("cx", 0.5) * W * S, p.get("cy", 0.5) * H * S
    r = p["diam"] * H * S / 2
    glow = Image.new("RGBA", big.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((cx - r * 1.5, cy - r * 1.5, cx + r * 1.5, cy + r * 1.5), fill=(200, 225, 255, 150))
    glow = glow.filter(ImageFilter.GaussianBlur(r * 0.55))
    big = Image.alpha_composite(big, glow)
    d = ImageDraw.Draw(big)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(232, 244, 255, 255))
    # 小人影：從正上方看，頭（紫）＋肩（白）
    d.ellipse((cx - r * 0.34, cy - r * 0.22, cx + r * 0.34, cy + r * 0.34), fill=(250, 250, 252, 255))
    d.ellipse((cx - r * 0.2, cy - r * 0.36, cx + r * 0.2, cy + r * 0.04), fill=(72, 36, 120, 255))
    dr = r / 2 * 0.5   # ECHO：直徑 1/4 圓盤直徑
    ex, ey = cx + r * 1.0, cy - r * 0.35
    d.ellipse((ex - dr, ey - dr, ex + dr, ey + dr), fill=(244, 244, 246, 255), outline=(168, 168, 176, 255), width=S // 2)
    d.rectangle((ex - dr * 0.7, ey - dr * 0.15, ex + dr * 0.7, ey + dr * 0.15), fill=(41, 211, 245, 255))
    return big.convert("RGB").resize((W, H), Image.LANCZOS)


def op_shift(im, p):
    """整幅平移（填邊用複製邊緣），用於微調構圖位置。dx、dy 為畫面比例。"""
    a = np.asarray(im)
    dx, dy = round(p.get("dx", 0) * W), round(p.get("dy", 0) * H)
    a = np.roll(a, (dy, dx), (0, 1))
    return Image.fromarray(a)


def op_zoom(im, p):
    """以 (cx,cy) 為中心放大 z 倍後裁回原尺寸（Qwen 的構圖偏寬時用）。"""
    z, (cx, cy) = p["z"], (p.get("cx", 0.5), p.get("cy", 0.5))
    cw, ch = W / z, H / z
    x0 = min(max(cx * W - cw / 2, 0), W - cw)
    y0 = min(max(cy * H - ch / 2, 0), H - ch)
    return im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((W, H), Image.LANCZOS)


def op_shrink_paste(im, p, ctx):
    """把 im 裡的大白色圓盤（含站在上面的人）縮成指定直徑，貼回 bg 圖（同 seed 的首幀 A，同樣的燈環）的正中。
    用於 s007 末幀：Qwen 會把平台畫成畫面高的 60%，但劇本要 20%；縮貼後首尾幀的燈環幾何也完全一致。"""
    from scipy import ndimage as ndi
    shot, frame, seed = ctx
    bg = Image.open(cdir(shot) / p["bg"].format(seed=seed)).convert("RGB").resize((W, H), Image.LANCZOS)
    a = np.asarray(im).astype(int)
    lab, n = ndi.label(a.min(axis=2) > p.get("white", 225))
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    ys, xs = np.where(lab == 1 + int(np.argmax(sizes)))
    d = max(xs.max() - xs.min(), ys.max() - ys.min())
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    box = (cx - 0.55 * d, cy - p.get("top", 0.68) * d, cx + 0.55 * d, cy + 0.55 * d)
    crop = im.crop(tuple(round(v) for v in box))
    f = p["target"] * H / d
    crop = crop.resize((round(crop.width * f), round(crop.height * f)), Image.LANCZOS)
    m = Image.new("L", crop.size, 0)
    md = ImageDraw.Draw(m)
    ccx, ccy, rr = (cx - box[0]) * f, (cy - box[1]) * f, 0.53 * d * f   # 只留圓盤本身＋頭頂（不要把 Qwen 畫在盤外的燈環帶進來）
    md.ellipse((ccx - rr, ccy - rr, ccx + rr, ccy + rr), fill=255)
    md.ellipse((ccx - 0.30 * d * f, ccy - 0.70 * d * f, ccx + 0.40 * d * f, ccy - 0.05 * d * f), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(d * f * 0.02))
    ox = round(W * p.get("cx", 0.5) - (cx - box[0]) * f)
    oy = round(H * p.get("cy", 0.5) - (cy - box[1]) * f)
    bg.paste(crop, (ox, oy), m)
    return bg


def op_glow(im, p):
    """放射狀柔光（screen 疊加）：平台白光照亮周圍牆面。cx、cy、r 為畫面比例（r 對畫面高）。"""
    yy, xx = np.mgrid[0:H, 0:W]
    r = p["r"] * H
    g = np.exp(-(((xx - p["cx"] * W) ** 2 + (yy - p["cy"] * H) ** 2) / (2 * (r / 2) ** 2))) * p.get("a", 0.5)
    col = np.array(p.get("color", [200, 225, 255]), float)
    base = np.asarray(im).astype(float)
    out = 255 - (255 - base) * (1 - g[..., None] * col / 255.0)
    return Image.fromarray(out.clip(0, 255).astype("uint8"))


def op_hatch(im, p):
    """@launch_roof 的艙口（程式繪製，位置與大小精確）：低角度下是一個扁橢圓。closed＝兩片半圓鋼板（接縫是一條朝向遠方的短直線）＋周圍淺色發光環；
    open＝鋼板已滑進地面下，橢圓洞口白光往上照（光柱＋光暈）。cx、cy＝橢圓中心（畫面比例），w＝橢圓寬（對畫面寬），ratio＝高／寬。"""
    S = 4
    cx, cy, w = p["cx"] * W, p["cy"] * H, p["w"] * W
    h = w * p.get("ratio", 0.16)
    pad = int(w * 1.4)
    x0, y0 = int(cx - w / 2 - pad), int(cy - h / 2 - pad * 1.2)
    bw, bh = int(w + 2 * pad), int(h + pad * 2.2)
    loc = Image.new("RGBA", (bw * S, bh * S), (0, 0, 0, 0))
    ex, ey = (cx - x0) * S, (cy - y0) * S
    rx, ry = w / 2 * S, h / 2 * S
    if p.get("state", "closed") == "closed":
        ring = Image.new("RGBA", loc.size, (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        k = 1.10
        rd.ellipse((ex - rx * k, ey - ry * k, ex + rx * k, ey + ry * k), outline=(236, 244, 246, 255), width=max(3, round(0.05 * w * S)))
        glow = ring.filter(ImageFilter.GaussianBlur(0.05 * w * S))
        loc = Image.alpha_composite(loc, glow)
        loc = Image.alpha_composite(loc, ring)
        d = ImageDraw.Draw(loc)
        d.ellipse((ex - rx, ey - ry, ex + rx, ey + ry), fill=(132, 140, 152, 255), outline=(30, 34, 42, 255), width=2 * S)
        d.ellipse((ex - rx * 0.93, ey - ry * 0.86, ex + rx * 0.93, ey + ry * 0.86), outline=(70, 76, 86, 255), width=S // 2)
        d.line([(ex, ey - ry), (ex, ey + ry)], fill=(20, 24, 30, 255), width=max(2 * S, round(0.02 * w * S)))   # 接縫：朝向遠方的短直線
    else:
        beam = Image.new("RGBA", loc.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(beam)
        up = p.get("beam", 0.12) * H * S
        for i in range(40):   # 往上漸淡的光柱
            t = i / 40
            bd.rectangle((ex - rx * (1 - 0.35 * t), ey - ry - up * (t + 0.025), ex + rx * (1 - 0.35 * t), ey - ry - up * t), fill=(225, 240, 255, int(150 * (1 - t) ** 1.5)))
        loc = Image.alpha_composite(loc, beam.filter(ImageFilter.GaussianBlur(0.05 * w * S)))
        gl = Image.new("RGBA", loc.size, (0, 0, 0, 0))
        ImageDraw.Draw(gl).ellipse((ex - rx * 1.3, ey - ry * 2.0, ex + rx * 1.3, ey + ry * 2.0), fill=(220, 238, 255, 190))
        loc = Image.alpha_composite(loc, gl.filter(ImageFilter.GaussianBlur(0.12 * w * S)))
        d = ImageDraw.Draw(loc)
        d.ellipse((ex - rx, ey - ry, ex + rx, ey + ry), fill=(240, 248, 255, 255), outline=(190, 200, 210, 255), width=S)
    loc = loc.resize((bw, bh), Image.LANCZOS)
    out = im.convert("RGBA")
    out.alpha_composite(loc, (x0, y0))
    return out.convert("RGB")


def detect_band(im, lo=0.25, hi=0.66, px=None):
    """彩色光帶的列範圍：飽和度最高那一群連續列（s008 首幀的地平線彩色街區）。回傳 (y0, y1) 像素。"""
    a = np.asarray(im).astype(float)
    sat = (a.max(axis=2) - a.min(axis=2)) / np.maximum(a.max(axis=2), 1) * (a.max(axis=2) / 255.0)   # 飽和度×亮度：霓虹帶亮、夜空暗
    r = sat.mean(axis=1)
    l, h = int(lo * H), int(hi * H)
    k = l + int(np.argmax(r[l:h]))
    if px:   # 只留最亮最飽和那一列上下各 px/2 像素：細的橫向光帶
        return max(l, k - px // 2), min(h - 1, k + px // 2)
    thr = 0.6 * r[k]
    y0 = y1 = k
    while y0 > l and r[y0 - 1] >= thr:
        y0 -= 1
    while y1 < h - 1 and r[y1 + 1] >= thr:
        y1 += 1
    return y0, y1


def op_floor_stretch(im, p):
    """Qwen 把樓頂地面畫成畫面下 20%（劇本：下 1/3）：偵測矮牆根（亮度梯度最大的列），上半部壓、地面部分拉，讓牆根落在 target。"""
    a = np.asarray(im.convert("L")).astype(float)
    r = a[:, 200:W - 200].mean(axis=1)
    g = np.abs(np.diff(r))
    lo, hi = int(p.get("lo", 0.6) * H), int(p.get("hi", 0.88) * H)
    wall = lo + int(np.argmax(g[lo:hi])) + 1
    t = round(p.get("target", 0.667) * H)
    top = im.crop((0, 0, W, wall)).resize((W, t), Image.LANCZOS)
    bot = im.crop((0, wall, W, H)).resize((W, H - t), Image.LANCZOS)
    out = Image.new("RGB", (W, H))
    out.paste(top, (0, 0))
    out.paste(bot, (0, t))
    return out


def op_band_grey(im, p):
    """首幀：整幅灰化，只留彩色光帶那幾列（自動偵測）；keep 的橢圓（LUNA 等）也保持彩色。"""
    y0, y1 = detect_band(im, p.get("lo", 0.25), p.get("hi", 0.66), p.get("band_px"))
    if p.get("boost"):   # 提亮彩色光帶那幾列（Qwen 畫成偏暗的藍紫帶；提亮後塔與樓的黑色剪影才看得清楚）
        a = np.asarray(im).astype(float)
        a[y0:y1 + 1] = np.minimum(a[y0:y1 + 1] * p["boost"], 255)
        im = Image.fromarray(a.astype("uint8"))
    where = [{"rect": [0, 0, 1, 1], "grey": True}, {"rect": [0, y0 / H, 1, (y1 + 1) / H], "grey": False, "feather": 1.5}] + list(p.get("keep", []))
    return op_grey(im, {"where": where})


def op_tower(im, p):
    """@signal_tower 的純黑剪影（程式繪製）：細塔身＋觀景台圓盤＋細頸＋天線與頂端白燈。cx＝中心（畫面比例）；
    base＝塔底的 y（畫面比例；'band' 表示彩色光帶的下緣）；h＝全高（對畫面高）。比例依 glossary：塔寬 10 m、觀景台 40 m、全高 240 m。"""
    S = 4
    cx = p.get("cx", 0.5) * W
    base = (detect_band(im, p.get("lo", 0.25), p.get("hi", 0.66))[1] if p.get("base") == "band" else p["base"] * H)
    h = p["h"] * H
    u = h / 240.0   # 每公尺的像素
    big = im.resize((W * S, H * S), Image.LANCZOS).convert("RGBA")
    d = ImageDraw.Draw(big)
    ink = (11, 13, 18, 255)
    x, b = cx * S, base * S
    def Y(m): return b - m * u * S
    if p.get("halo", True):   # 塔後的霓虹低雲光暈：讓黑色剪影讀得出來（末幀描述：邊緣被映著霓虹的低雲勾出細亮邊）
        hl = Image.new("RGBA", big.size, (0, 0, 0, 0))
        ImageDraw.Draw(hl).ellipse((x - 26 * u * S, Y(215), x + 26 * u * S, Y(40)), fill=(150, 130, 220, 150))
        big = Image.alpha_composite(big, hl.filter(ImageFilter.GaussianBlur(9 * u * S)))
        d = ImageDraw.Draw(big)
    d.rectangle((x - 5 * u * S, Y(154), x + 5 * u * S, b), fill=ink)                                   # 塔身
    d.ellipse((x - 20 * u * S, Y(166), x + 20 * u * S, Y(154)), fill=ink)                                # 觀景台
    d.rectangle((x - 20 * u * S, Y(160), x + 20 * u * S, Y(158)), fill=ink)
    d.polygon([(x - 3 * u * S, Y(166)), (x + 3 * u * S, Y(166)), (x + 1 * u * S, Y(180)), (x - 1 * u * S, Y(180))], fill=ink)   # 細頸
    d.rectangle((x - 0.8 * u * S, Y(240), x + 0.8 * u * S, Y(178)), fill=ink)                            # 天線
    if p.get("rim", True):   # 背後低雲映霓虹：左右緣一道極細的亮邊（glossary／末幀描述）
        rim = (88, 104, 150, 255)
        d.line([(x - 5 * u * S, Y(154)), (x - 5 * u * S, b)], fill=rim, width=max(1, S // 2))
        d.line([(x + 5 * u * S, Y(154)), (x + 5 * u * S, b)], fill=rim, width=max(1, S // 2))
    r = max(S, 2.2 * u * S)
    d.ellipse((x - r, Y(240) - r, x + r, Y(240) + r), fill=(244, 248, 255, 255))                          # 頂端白燈
    return big.convert("RGB").resize((W, H), Image.LANCZOS)


def op_roof(im, p, seed=0):
    """s008 首幀：低機位（1.2 m 平視）的樓頂：矮牆橫過畫面、地面在畫面下 1/3（水泥板縫往地平線收斂、幾攤水），再加停在半空的雨滴（小圓點）。
    im 是只有城市天際線的全景，牆與地面蓋在它下方。程式繪製、平塗＋細線，風格貼近 refs/entities/launch_roof。"""
    S = 2
    rnd = np.random.RandomState(p.get("seed", 7) + seed)
    base = im.resize((W * S, H * S), Image.LANCZOS).convert("RGBA")
    d = ImageDraw.Draw(base)
    wt, wb = p.get("wall_top", 0.605) * H * S, p.get("wall_bottom", 0.667) * H * S
    ink = (18, 22, 30, 255)
    # 地面：由遠到近漸暗
    for y in range(int(wb), H * S):
        t = (y - wb) / (H * S - wb)
        c = (int(58 - 22 * t), int(66 - 24 * t), int(84 - 28 * t), 255)
        d.line([(0, y), (W * S, y)], fill=c)
    vp = (W * S / 2, p.get("vp_y", 0.52) * H * S)
    for k in range(-14, 15):   # 往地平線收斂的板縫
        x_bot = W * S / 2 + k * 330 * S
        t0 = (wb - vp[1]) / (H * S - vp[1])
        x_top = vp[0] + (x_bot - vp[0]) * t0
        d.line([(x_top, wb), (x_bot, H * S)], fill=(30, 36, 48, 255), width=2 * S)
    for i in range(1, 6):      # 橫向板縫（愈近愈疏）
        y = wb + (H * S - wb) * (i / 5.0) ** 1.8
        d.line([(0, y), (W * S, y)], fill=(30, 36, 48, 255), width=2 * S)
    for _ in range(int(p.get("puddles", 9))):   # 水窪：扁橢圓，愈近愈大
        t = rnd.uniform(0.08, 0.95)
        y = wb + (H * S - wb) * t
        w = (40 + 360 * t) * S * rnd.uniform(0.6, 1.2)
        x = rnd.uniform(0.05, 0.95) * W * S
        d.ellipse((x - w / 2, y - w * 0.07, x + w / 2, y + w * 0.07), fill=(86, 100, 124, 255), outline=ink, width=S)
        d.arc((x - w * 0.4, y - w * 0.05, x + w * 0.4, y + w * 0.05), 200, 330, fill=(150, 168, 196, 255), width=S)
    # 矮牆：正面＋頂面、接縫
    d.rectangle((0, wt, W * S, wb), fill=(66, 74, 92, 255))
    d.rectangle((0, wt, W * S, wt + 7 * S), fill=(98, 108, 128, 255))
    d.line([(0, wt), (W * S, wt)], fill=ink, width=2 * S)
    d.line([(0, wb), (W * S, wb)], fill=ink, width=2 * S)
    d.line([(0, wt + 7 * S), (W * S, wt + 7 * S)], fill=(40, 46, 58, 255), width=S)
    for k in range(0, 10):
        x = (k * 150 + 40) * S
        d.line([(x, wt + 7 * S), (x, wb)], fill=(44, 50, 64, 255), width=S)
    # 停在半空的雨滴：小圓點（不是雨線）
    drops = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(drops)
    for _ in range(p.get("drops", 420)):
        x, y = rnd.uniform(0, W * S), (rnd.uniform(0, 1) ** 0.65) * H * S   # 整個畫面都有，偏向近處（下方）；近大遠小
        r = rnd.uniform(0.9, 2.0) * (1 + 1.8 * y / (H * S)) * S
        dd.ellipse((x - r, y - r, x + r, y + r), fill=(176, 184, 196, int(rnd.uniform(110, 190))))
    base = Image.alpha_composite(base, drops)
    return base.convert("RGB").resize((W, H), Image.LANCZOS)


SPRITE = (832, 1216)


def step_sprite(shot, frame, seed, v):
    """小人物用：先在全尺寸（832×1216）以 Illustrious＋LoRA 加 OpenPose 骨架（p3_lora_gen.add_controlnet）出單人立繪（白底），
    摳圖後在後製 paste_sprite 縮小貼進場景。比起讓 Qwen／B 在 15% 畫面高的小人物上重畫，姿勢、五官、服裝都清楚得多。"""
    sp = v["sprite"]
    d = cdir(shot)
    tags = sp["tags"]
    pos = L.BASE + L.prompt_outfit(tags) + tags + L.QUALITY
    wf = L.workflow(K.LORA, 0.9, pos, SPRITE[0], SPRITE[1], seed, 28, 6.0)
    pose = sp.get("pose")
    rel = None
    if isinstance(pose, dict):
        pp = d / f"{frame}_pose_s{seed}.png"
        draw_pose(pose, (0, 0, *SPRITE), SPRITE, pp, canvas=SPRITE)
        rel = str(pp.relative_to(ROOT))
    else:
        rel = pose
    L.add_controlnet(wf, rel, sp.get("lineart"), sp.get("lineart_strength", 0.6))
    for k in wf:
        if wf[k].get("class_type") == "ControlNetApplyAdvanced":
            is_pose = wf[wf[k]["inputs"]["control_net"][0]]["inputs"]["type"] == "openpose"
            wf[k]["inputs"]["end_percent"] = sp.get("pose_end", 0.8) if is_pose else sp.get("lineart_end", 0.6)
            wf[k]["inputs"]["strength"] = sp.get("pose_strength", 1.0) if is_pose else sp.get("lineart_strength", 0.6)
    t0 = time.time()
    data = K.run(wf, "9")
    fs = d / f"{frame}_sprite_s{seed}.png"
    fs.write_bytes(data)
    cut = cutout_sprite(fs)
    if seed in (v.get("flip") or []):   # LoRA 畫背面時腿環常落在畫面左腿（角色左腿）：只水平翻轉這個立繪（立繪裡其他部分左右對稱），讓腿環在畫面右＝角色右腿
        cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
    cut.save(d / f"{frame}_cut_s{seed}.png")
    log(shot, dict(step="B", frame=frame, file=fs.name, seed=seed, denoise=1.0, pre=None, controlnet=dict(pose=rel, pose_strength=sp.get("pose_strength", 1.0),
                   pose_end=sp.get("pose_end", 0.8), canny=False), region=None, prompt=pos, seconds=round(time.time() - t0, 1)))
    return fs


def cutout_sprite(path):
    """白底立繪摳圖：由四邊洪水填充去掉近白背景（黑色輪廓線擋住白衣），其餘為前景。"""
    from scipy import ndimage as ndi
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bgc = np.median(border, axis=0)                      # 立繪背景可能是白或淺灰，取四邊的中位色
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    lum = a.mean(axis=2)
    near = (np.abs(a - bgc).max(axis=2) < 26) | ((sat < 0.14) & (lum > 95) & (lum < 222))   # 背景色，加上背景的灰色投影（低飽和中灰）
    lab, n = ndi.label(near)
    bl = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    fg = ndi.binary_fill_holes(~np.isin(lab, list(bl)))
    fg = ndi.binary_opening(fg, iterations=2)
    lab2, n2 = ndi.label(fg)
    if n2:
        sizes = ndi.sum(fg, lab2, range(1, n2 + 1))
        fg = lab2 == (1 + int(np.argmax(sizes)))
    m = Image.fromarray((fg * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.0))
    rgba = im.convert("RGBA")
    rgba.putalpha(m)
    return rgba.crop(rgba.getchannel("A").point(lambda x: 255 if x > 40 else 0).getbbox())


def op_paste_sprite(im, p, ctx):
    """把摳好的立繪縮成目標身高（畫面高比例）貼到腳點 (fx, fy)，腳下加一個柔和的接觸陰影。"""
    shot, frame, seed = ctx
    cut = Image.open(cdir(shot) / f"{frame}_cut_s{seed}.png").convert("RGBA")
    th = p["target_h"] * H
    f = th / cut.height
    cut = cut.resize((max(1, round(cut.width * f)), round(th)), Image.LANCZOS)
    out = im.convert("RGBA")
    fx, fy = p["fx"] * W, p["fy"] * H
    sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
    sw = cut.width * 0.9
    ImageDraw.Draw(sh).ellipse((fx - sw / 2, fy - th * 0.05, fx + sw / 2, fy + th * 0.05), fill=(0, 0, 0, 120))
    out = Image.alpha_composite(out, sh.filter(ImageFilter.GaussianBlur(th * 0.03)))
    out.alpha_composite(cut, (round(fx - cut.width * p.get("anchor_x", 0.5)), round(fy - cut.height)))
    return out.convert("RGB")


def op_skyline_from(im, p, ctx):
    """s008 首幀：取同 seed 末幀 Qwen 場景（last_A）矮牆以上的城市，縱向拉伸成平視背景的上半（牆頂在 wall_top）。
    矮牆頂 ＝ 亮度梯度最大的列（在 lo–hi 之間）。這樣首尾幀的城市是同一座。"""
    shot, frame, seed = ctx
    a = Image.open(cdir(shot) / p["src"].format(seed=seed)).convert("RGB").resize((W, H), Image.LANCZOS)
    g = np.abs(np.diff(np.asarray(a.convert("L")).astype(float)[:, 200:W - 200].mean(axis=1)))
    lo, hi = int(p.get("lo", 0.45) * H), int(p.get("hi", 0.68) * H)
    wall = lo + int(np.argmax(g[lo:hi]))
    t = round(p.get("wall_top", 0.605) * H)
    top = a.crop((0, 0, W, wall)).resize((W, t), Image.LANCZOS)
    out = Image.new("RGB", (W, H), (30, 34, 44))
    out.paste(top, (0, 0))
    return out


OPS = dict(skyline_from=op_skyline_from, roof=op_roof, floor_stretch=op_floor_stretch, band_grey=op_band_grey, tower=op_tower, grey=op_grey, glow=op_glow, hatch=op_hatch, paste_drone=op_paste_drone, platform_dot=op_platform_dot, shift=op_shift, zoom=op_zoom)


def step_C(shot, frame, seed, v, src):
    im = Image.open(src).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    im, done = apply_ops(im, v.get("post"), (shot, frame, seed))
    im.save(cdir(shot) / f"{frame}_Cpre_s{seed}.png")   # 裁切前的 1344×768（mid 以首幀的這張當場景，避免二次裁切）
    # 1344×768 → 1280×720：縮放涵蓋後置中裁切
    s = max(OUT_W / im.width, OUT_H / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - OUT_W) // 2, (im.height - OUT_H) // 2
    im = im.crop((l, t, l + OUT_W, t + OUT_H))
    ff = fF(shot, frame, seed)
    im.save(ff)
    log(shot, dict(step="C", frame=frame, file=ff.name, seed=seed, post=done, size=f"{OUT_W}x{OUT_H}"))
    return ff


# ---------------------------------------------------------------- 一幀一 seed
def variants(fj):
    """frame job → [(variant dict, seeds)]；沒有 variants 就是一個，seeds 取自 job。"""
    base = {k: val for k, val in fj.items() if k != "variants"}
    if "variants" not in fj:
        return [(base, fj.get("seeds", []))]
    out = []
    for var in fj["variants"]:
        v = dict(base)
        v.update(var)
        out.append((v, var["seeds"]))
    return out


def run_one(shot, frame, seed, v, macros, stage, force):
    ff, fb, fa = fF(shot, frame, seed), fB(shot, frame, seed), fA(shot, frame, seed)
    gen = v.get("gen", "qwen")
    if gen == "proc" and not fa.exists():
        Image.new("RGB", (W, H), (4, 5, 7)).save(fa)   # 程式繪製的候選：A 只是占位，內容由 post 的 op 畫
    if gen == "sprite":
        # 場景：scene=first → 同 seed 首幀的裁切前成品；否則 Qwen 出（不含人物）。人物＝立繪（B 加骨架）摳圖後在 post 貼
        if (stage in ("all", "A") and (force or not fa.exists())) or (v.get("scene") == "first" and (not fa.exists() or stage in ("post", "cut", "BC"))):
            if v.get("scene") == "first":
                ss = v.get("srcseed", (seed - 1) % 6 + 1)   # mid 的 seed 7–12 沿用首幀 seed 1–6 的場景
                shutil.copy(cdir(shot) / f"first_Cpre_s{ss}.png", fa)
                log(shot, dict(step="A", frame=frame, file=fa.name, seed=seed, refs=[f"first_Cpre_s{ss}.png"],
                               prompt="(沿用同 seed 首幀的場景，不經 Qwen)", seconds=0))
            else:
                step_A(shot, frame, seed, v, macros, None)
        if stage in ("all", "B", "BC") and (force or not (cdir(shot) / f"{frame}_cut_s{seed}.png").exists()):
            step_sprite(shot, frame, seed, v)
        if stage == "cut":   # 只從已存的立繪重新摳圖（改摳圖演算法或 flip 清單後用），再做後製
            cut = cutout_sprite(cdir(shot) / f"{frame}_sprite_s{seed}.png")
            if seed in (v.get("flip") or []):
                cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
            cut.save(cdir(shot) / f"{frame}_cut_s{seed}.png")
        if stage in ("all", "post", "BC", "cut") and (force or stage in ("post", "BC", "cut") or not ff.exists()):
            step_C(shot, frame, seed, v, fa)
        return ff
    if stage in ("all", "A") and gen == "qwen" and (force or not fa.exists()):
        step_A(shot, frame, seed, v, macros, None)
    use_b = v.get("lora", True)
    if stage in ("all", "B", "BC") and use_b and (force or not fb.exists()):
        step_B(shot, frame, seed, v, fa)
    if stage in ("all", "post", "BC") and (force or stage in ("post", "BC") or not ff.exists()):
        step_C(shot, frame, seed, v, fb if use_b else fa)
    return ff


# ---------------------------------------------------------------- 對照圖
def contact(shot, frame):
    rj = rejects(shot).get(frame, {})
    pat = re.compile(rf"^{frame}_s(\d+)$")   # 只收最終候選 <frame>_s<N>.png（不含 _A_／_B_／_sprite_／_cut_ 等中間稿）
    files = sorted((f for f in cdir(shot).glob(f"{frame}_s*.png") if pat.match(f.stem) and pat.match(f.stem).group(1) not in rj),
                   key=lambda f: int(pat.match(f.stem).group(1)))
    if not files:
        return None
    cw, chh = 672, 378
    n = len(files)
    cols = 2
    ins, filt, names = [], [], []
    for i, f in enumerate(files):
        ins += ["-i", str(f)]
        lab = f"{frame} #{f.stem.split('_s')[-1]}"
        filt.append(f"[{i}:v]scale={cw}:{chh},drawtext=fontfile={FONT}:text='{lab}':x=8:y=8:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6[v{i}]")
        names.append(f"v{i}")
    layout = "|".join(f"{(i % cols) * cw}_{(i // cols) * chh}" for i in range(n))
    fc = ";".join(filt) + ";" + ("[v0]null[out]" if n == 1 else "".join(f"[{x}]" for x in names) + f"xstack=inputs={n}:layout={layout}:fill=black[out]")
    out = ROOT / "shots" / shot / f"contact_{frame}.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", fc, "-map", "[out]", "-frames:v", "1", str(out)], check=True)
    return out


# ---------------------------------------------------------------- keyframe.json
def record(shot, jobs):
    res = json.loads((cdir(shot) / "results.json").read_text()) if (cdir(shot) / "results.json").exists() else []
    rj = rejects(shot)
    out = {"pick": "", "note": "pick 欄由使用者挑後填：keyframe.png／last.png／mid.png = candidates/<frame>_s<N>.png（最終候選皆為 1280×720）"}
    for frame, fj in jobs["shots"][shot]["frames"].items():
        last = {}
        for r in res:
            if r.get("frame") == frame:
                last.setdefault(r["seed"], {})[r["step"]] = r   # 同 seed 重跑取最新
        entry = {"說明": fj.get("desc", ""), "variants": []}
        for v, seeds in variants(fj):
            seeds_ok = [s for s in seeds if s in last and "C" in last[s]]
            if not seeds_ok:
                continue
            s0 = last[seeds_ok[0]]
            ve = {"seeds": seeds_ok, "gen": v.get("gen", "qwen")}
            if "A" in s0:
                ve["A"] = dict(refs=s0["A"]["refs"], prompt=s0["A"]["prompt"])
            if "B" in s0:
                ve["B"] = dict(denoise=s0["B"]["denoise"], pre=s0["B"].get("pre"), controlnet=s0["B"]["controlnet"], region=s0["B"]["region"], prompt=s0["B"]["prompt"])
            ve["post"] = s0["C"]["post"]
            ve["seconds_per_candidate"] = {str(s): round(sum(x.get("seconds", 0) for x in last[s].values()), 1) for s in seeds_ok}
            ve["files"] = {str(s): last[s]["C"]["file"] for s in seeds_ok}
            entry["variants"].append(ve)
        if frame in rj:
            entry["rejected"] = rj[frame]
        entry["contact"] = f"contact_{frame}.png"
        out[frame] = entry
    p = ROOT / "shots" / shot / "keyframe.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return p


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="")
    ap.add_argument("--section", default="")
    ap.add_argument("--frames", default="")
    ap.add_argument("--seeds", default="", help="只跑這些 seed（預設用 jobs 檔）")
    ap.add_argument("--stage", default="all", choices=["all", "A", "B", "BC", "post", "cut"])
    ap.add_argument("--force", action="store_true", help="已存在的檔案也重做")
    ap.add_argument("--contact", action="store_true", help="只重拼對照圖")
    ap.add_argument("--record", action="store_true", help="只重寫 keyframe.json")
    ap.add_argument("--reject", default="", help="frame:seed:原因（剔除候選，對照圖不放；seed 為空則取消剔除用 frame:seed:）")
    a = ap.parse_args()
    jobs = load_jobs()
    macros = jobs.get("macros", {})
    shots = load_shots()
    sel = [s for s in a.shots.split(",") if s]
    if a.section:
        sel += [i for i in jobs["shots"] if shots[i]["section"] == a.section and i not in sel]
    if not sel:
        sel = list(jobs["shots"])
    for shot in sel:
        if a.reject:
            fr, sd, why = a.reject.split(":", 2)
            rj = rejects(shot)
            if why:
                rj.setdefault(fr, {})[sd] = why
            else:
                rj.get(fr, {}).pop(sd, None)
            (cdir(shot) / "rejects.json").write_text(json.dumps(rj, ensure_ascii=False, indent=1))
        if not (a.contact or a.record or a.reject):
            for frame, fj in jobs["shots"][shot]["frames"].items():
                if a.frames and frame not in a.frames.split(","):
                    continue
                for v, seeds in variants(fj):
                    for sd in (int(x) for x in a.seeds.split(",")) if a.seeds else seeds:
                        if a.seeds and sd not in seeds:
                            continue
                        t0 = time.time()
                        f = run_one(shot, frame, sd, v, macros, a.stage, a.force)
                        print(shot, frame, sd, f.name, f"{time.time() - t0:.0f}s", flush=True)
        for frame in jobs["shots"][shot]["frames"]:
            if not a.frames or frame in a.frames.split(","):
                c = contact(shot, frame)
                print("contact", c, flush=True)
        print("keyframe.json", record(shot, jobs), flush=True)


if __name__ == "__main__":
    main()
