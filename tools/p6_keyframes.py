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
        for fr in FRAMES:   # {first}／{last}／{mid}＝同鏡該幀的 LoRA／A 中間稿（有 B 用 B，沒有就用 A）；{first:final}＝最終稿；{first:A}＝Qwen 稿
            for suffix, kind in (("{%s:final}" % fr, "final"), ("{%s:A}" % fr, "A"), ("{%s:Bin}" % fr, "Bin"), ("{%s}" % fr, "mid")):
                if suffix in m:
                    sd = v.get("srcseed", seed)
                    binf = cdir(shot) / f"{fr}_Bin_s{sd}.png"   # B 的輸入稿（A 經 pre 構圖修正後）
                    cand = (fF(shot, fr, sd) if kind == "final" else fA(shot, fr, sd) if kind == "A" else (binf if binf.exists() else fA(shot, fr, sd)) if kind == "Bin"
                            else (fB(shot, fr, sd) if fB(shot, fr, sd).exists() else fA(shot, fr, sd)))
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
        elif op == "grey_keep":
            im = op_grey_keep(im, st, ctx)
        elif op == "paste_asset":
            im = op_paste_asset(im, st, ctx)
        elif op == "paste_head":
            im = op_paste_head(im, st, ctx)
        elif op in ("frozen_rain", "billboards", "harmonize"):
            im = OPS[op](im, st, ctx)
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
    if v.get("neg_add"):   # 例：凍結的雨不要雨絲、眼睛不要棕紅
        wf["5"]["inputs"]["text"] = L.NEG + ", " + v["neg_add"]
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


def sky_mask_cols(im, y0=0.2, tol=14, top=None):
    """天空遮罩（逐欄掃描）：每一欄從 y0 往下掃，第一次和天空底色（最小值濾波去掉雨絲後）差超過 tol 就是天際線；以上都是天空。
    用在暗藍天空和更暗樓群亮度很接近、連通區法會把樓群也算進去的場景（s014）。"""
    from scipy import ndimage as ndi
    L = np.asarray(im.convert("L").filter(ImageFilter.MinFilter(5))).astype(float)
    r0 = round(y0 * H)
    m = np.zeros((H, W), bool)
    ref = np.median(L[r0:r0 + 12], axis=0)
    dev = np.abs(L - ref[None, :]) > tol
    dev[:r0] = False
    dev2 = dev[:-1] & dev[1:]
    first = np.where(dev2.any(axis=0), dev2.argmax(axis=0), H)
    first = ndi.median_filter(first, size=9)
    rt = round((top if top is not None else y0) * H)
    for x in range(W):
        m[rt:max(rt, first[x]), x] = True
    return Image.fromarray((m * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.8))


def sky_mask(im, cx, cy, thr=80):
    """天空遮罩：含 (cx,cy) 的暗色連通區（閉運算後，細的雨滴不切斷它）。"""
    from scipy import ndimage as ndi
    a = np.asarray(im.convert("L")).astype(float)
    dark = ndi.binary_closing(a < thr, structure=np.ones((3, 3)), iterations=5)
    lab, _ = ndi.label(dark)
    k = lab[round(cy * im.height), round(cx * im.width)]
    m = (lab == k) if k else np.zeros_like(dark)
    return Image.fromarray((m * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.8))


def op_tower(im, p):
    """@signal_tower 的純黑剪影（程式繪製）：細塔身＋觀景台圓盤＋細頸＋天線與頂端白燈。cx＝中心（畫面比例）；
    base＝塔底的 y（畫面比例；'band' 表示彩色光帶的下緣）；h＝全高（對畫面高）。比例依 glossary：塔寬 10 m、觀景台 40 m、全高 240 m。
    behind_sky＝只畫在天空（含 (cx,sky_y) 的暗色連通區）上，被樓群擋住；fat＝橫向加粗倍數。"""
    if p.get("behind_sky"):
        q = dict(p); q["behind_sky"] = False
        full = op_tower(im, q)
        sm = sky_mask_cols(im, p.get("sky_y", 0.2), p.get("sky_tol", 14), p.get("sky_top")) if p.get("sky_mode") == "cols" else sky_mask(im, p.get("cx", 0.5), p.get("sky_y", 0.28), p.get("sky_thr", 80))
        return Image.composite(full, im, sm)
    S = 4
    cx = p.get("cx", 0.5) * W
    base = (detect_band(im, p.get("lo", 0.25), p.get("hi", 0.66))[1] if p.get("base") == "band" else p["base"] * H)
    h = p["h"] * H
    u = h / 240.0   # 每公尺的像素
    ux = u * p.get("fat", 1.0)   # 橫向加粗倍數（塔很小時剪影讀不出來，略加粗並在回報說明）
    big = im.resize((W * S, H * S), Image.LANCZOS).convert("RGBA")
    d = ImageDraw.Draw(big)
    ink = (11, 13, 18, 255)
    x, b = cx * S, base * S
    def Y(m): return b - m * u * S
    if p.get("halo", True):   # 塔後的霓虹低雲光暈：讓黑色剪影讀得出來（末幀描述：邊緣被映著霓虹的低雲勾出細亮邊）
        hl = Image.new("RGBA", big.size, (0, 0, 0, 0))
        ImageDraw.Draw(hl).ellipse((x - 26 * ux * S, Y(215), x + 26 * ux * S, Y(40)), fill=(150, 130, 220, p.get("halo_a", 150)))
        big = Image.alpha_composite(big, hl.filter(ImageFilter.GaussianBlur(9 * u * S)))
        d = ImageDraw.Draw(big)
    d.rectangle((x - 5 * ux * S, Y(154), x + 5 * ux * S, b), fill=ink)                                   # 塔身
    d.ellipse((x - 20 * ux * S, Y(166), x + 20 * ux * S, Y(154)), fill=ink)                                # 觀景台
    d.rectangle((x - 20 * ux * S, Y(160), x + 20 * ux * S, Y(158)), fill=ink)
    d.polygon([(x - 3 * ux * S, Y(166)), (x + 3 * ux * S, Y(166)), (x + 1 * ux * S, Y(180)), (x - 1 * ux * S, Y(180))], fill=ink)   # 細頸
    d.rectangle((x - 0.8 * ux * S, Y(240), x + 0.8 * ux * S, Y(178)), fill=ink)                            # 天線
    if p.get("rim", True):   # 背後低雲映霓虹：左右緣一道極細的亮邊（glossary／末幀描述）
        rim = (88, 104, 150, 255)
        d.line([(x - 5 * ux * S, Y(154)), (x - 5 * ux * S, b)], fill=rim, width=max(1, S // 2))
        d.line([(x + 5 * ux * S, Y(154)), (x + 5 * ux * S, b)], fill=rim, width=max(1, S // 2))
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


def person_mask(im):
    """前景（LUNA 等）遮罩：ComfyUI 的 BiRefNet（RemoveBackground）。回傳與 im 同尺寸的 L 圖（255＝前景）。"""
    import os
    tmp = ROOT / "shots" / f".p6_tmp_mask_{os.getpid()}.png"
    im.save(tmp)
    name = K.put(tmp, f"p6_mask_in_{os.getpid()}.png")
    tmp.unlink()
    wf = {"1": {"class_type": "LoadImage", "inputs": {"image": name}},
          "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
          "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
          "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
          "9": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "p6mask/x"}}}
    data = K.run(wf, "9")
    out = ROOT / "shots" / f".p6_tmp_mask_out_{os.getpid()}.png"
    out.write_bytes(data)
    m = Image.open(out).convert("L").resize(im.size, Image.LANCZOS)
    out.unlink()
    return m


def _one_component(m, seed=None):
    """遮罩只留一個連通區：含 seed（畫面比例 [x,y]）的那個，沒給就取最大的（BiRefNet 會把氣球、小孩也圈進來）。"""
    from scipy import ndimage as ndi
    a = np.asarray(m) > 0
    lab, n = ndi.label(a)
    if n <= 1:
        return m
    if seed:
        k = lab[round(seed[1] * m.height), round(seed[0] * m.width)]
    else:
        k = 0
    if not k:
        k = 1 + int(np.argmax(ndi.sum(a, lab, range(1, n + 1))))
    return Image.fromarray(((lab == k) * 255).astype("uint8"))


def hue_mask(im, ranges):
    """依色相保留的遮罩（紅氣球、黃雨衣等）：ranges＝[{h:[h0,h1] 度, s: 最低飽和度, v: 最低亮度, rect: [x0,y0,x1,y1] 比例（可省略）}]。"""
    import cv2
    hsv = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2HSV).astype(float)   # H 0–179
    hh, ss, vv = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
    out = np.zeros(im.size[::-1], bool)
    for r in ranges:
        h0, h1 = r["h"]
        inh = ((hh >= h0) & (hh <= h1)) if h0 <= h1 else ((hh >= h0) | (hh <= h1))
        sel = inh & (ss >= r.get("s", 0.35)) & (vv >= r.get("v", 0.25))
        if r.get("rect"):
            box = np.zeros_like(sel)
            x0, y0, x1, y1 = r["rect"]
            box[round(y0 * im.height):round(y1 * im.height), round(x0 * im.width):round(x1 * im.width)] = True
            sel &= box
        out |= sel
    m = Image.fromarray((out * 255).astype("uint8"))
    from scipy import ndimage as ndi
    m = Image.fromarray((ndi.binary_closing(np.asarray(m) > 0, iterations=3) * 255).astype("uint8"))
    return m.filter(ImageFilter.MaxFilter(5))


def stash_aux(shot, frame, seed, name, im):
    """把中間狀態存成 candidates/<frame>_<name>_s<seed>.png（1280×720，與候選同樣的縮放裁切；給路線 C 後製用）。"""
    s = max(OUT_W / im.width, OUT_H / im.height)
    r = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (r.width - OUT_W) // 2, (r.height - OUT_H) // 2
    r.crop((l, t, l + OUT_W, t + OUT_H)).save(cdir(shot) / f"{frame}_{name}_s{seed}.png")


def op_grey_keep(im, p, ctx=None):
    """frozen 灰化，但保留彩色的部分：person＝BiRefNet 的前景遮罩（LUNA，可 dilate 擴張 px、feather 羽化）；
    keep＝額外保留的矩形／橢圓（同 grey 的 where 格式，grey 填 false）；hue＝依色相保留（紅氣球、黃雨衣）。三者聯集保持彩色，其餘去飽和。
    mask_img＝存下遮罩到 candidates 方便檢查（給檔名）。"""
    keep = Image.new("L", im.size, 0)
    if p.get("person", True):
        pm = person_mask(im)
        pm = pm.point(lambda x: 255 if x > p.get("thr", 100) else 0)
        pm = _one_component(pm, p.get("person_seed"))
        if p.get("dilate", 6):
            pm = pm.filter(ImageFilter.MaxFilter(p.get("dilate", 6) * 2 + 1))
        keep = Image.fromarray(np.maximum(np.asarray(keep), np.asarray(pm)))
    if p.get("hue"):
        keep = Image.fromarray(np.maximum(np.asarray(keep), np.asarray(hue_mask(im, p["hue"]))))
    if p.get("keep_head") and _HEAD.get("mask") is not None:   # 剛貼上的 LUNA 後腦肩膀（模糊後的 alpha）
        keep = Image.fromarray(np.maximum(np.asarray(keep), np.asarray(_HEAD["mask"].point(lambda x: 255 if x > 20 else 0))))
    for w in p.get("keep", []):
        shape = w.get("rect") or w.get("ellipse")
        box = [shape[0] * im.width, shape[1] * im.height, shape[2] * im.width, shape[3] * im.height]
        tmp = Image.new("L", im.size, 0)
        (ImageDraw.Draw(tmp).rectangle if w.get("rect") else ImageDraw.Draw(tmp).ellipse)(box, fill=255)
        if w.get("feather", 0):
            tmp = tmp.filter(ImageFilter.GaussianBlur(w["feather"]))
        keep = Image.fromarray(np.maximum(np.asarray(keep), np.asarray(tmp)))
    keep = keep.filter(ImageFilter.GaussianBlur(p.get("feather", 1.2)))
    grey = ImageOps.grayscale(im).convert("RGB")
    frozen = Image.composite(im, grey, keep)
    if p.get("split"):   # 只灰化斜界線左邊（路線 C：灰彩交界）：界線在畫面高 50% 處的 x＝split.x，tilt＝每單位畫面高的 x 位移
        sp = p["split"]
        yy, xx = np.mgrid[0:im.height, 0:im.width]
        xb = (sp["x"] + sp.get("tilt", 0.0) * (yy / im.height - 0.5)) * im.width
        left = Image.fromarray(((xx < xb) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(sp.get("soft", 0.8)))
        out = Image.composite(frozen, im, left)
    else:
        out = frozen
    if p.get("stash") and ctx:
        shot, frame, seed = ctx
        for nm in p["stash"]:
            stash_aux(shot, frame, seed, nm, {"color": im, "frozen": frozen}[nm])
    return out


_DRONE_TEX = {}


def drone_sprite(diam_px, yaw=0.0, S=4):
    """ECHO 的球體貼圖旋轉（front.png＋back.png 當前後半球的貼圖，正交投影）：yaw 度數，正值＝黑色面罩與光帶往畫面右轉；
    0＝正面、180＝背面。不能整顆重畫（G0：球體轉不過去），所以用球面映射轉。回傳 RGBA（直徑 diam_px）。"""
    import cv2
    if not _DRONE_TEX:
        for k in ("front", "back"):
            c = np.asarray(P4.cutout_drone(f"refs/entities/drone/{k}.png"))
            _DRONE_TEX[k] = c
    n = max(8, round(diam_px * S))
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    R = n / 2
    u, v = (xx - R + 0.5) / R, (yy - R + 0.5) / R
    r2 = u * u + v * v
    inside = r2 <= 1.0
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    th = math.radians(yaw)
    n0x = u * math.cos(th) - z * math.sin(th)
    n0z = u * math.sin(th) + z * math.cos(th)
    out = np.zeros((n, n, 4), np.float32)
    wf = np.clip((n0z + 0.14) / 0.28, 0, 1)[..., None]   # 前後半球貼圖在邊界附近混合，避開貼圖外緣的黑輪廓
    for k, sgn, wt in (("front", 1, wf), ("back", -1, 1 - wf)):
        tex = _DRONE_TEX[k]
        th_, tw_ = tex.shape[:2]
        rr = min(th_, tw_) / 2 * 0.95
        mx = np.clip(sgn * n0x, -1, 1) * rr + tw_ / 2
        my = np.clip(v, -1, 1) * rr + th_ / 2
        samp = cv2.remap(tex, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        out += samp * wt
    out[..., 3] = np.where(inside, 255, 0)
    im = Image.fromarray(out.clip(0, 255).astype("uint8"), "RGBA")
    ol = Image.new("RGBA", im.size, (0, 0, 0, 0))   # 外緣的細黑輪廓
    ImageDraw.Draw(ol).ellipse((0, 0, n - 1, n - 1), outline=(18, 22, 30, 255), width=max(2, round(n * 0.014)))
    im = Image.alpha_composite(im, ol)
    return im.resize((max(2, round(diam_px)), max(2, round(diam_px))), Image.LANCZOS)


def op_drone(im, p):
    """貼 ECHO（可轉向）：cx、cy＝球心（畫面比例）、diam＝直徑（對畫面高）、yaw＝轉向度數（見 drone_sprite）；
    behind＝true 時用 BiRefNet 的前景遮罩把 LUNA 蓋回 ECHO 上面（ECHO 在她肩後被擋住一部分）。"""
    p = dict(p)
    if p.get("rel"):   # 依 LUNA 的前景遮罩定位：頭高＝遮罩高/7.5（全身）或指定（對畫面高）；dx、dy＝球心相對頭頂中心（單位：頭高；dx 正＝畫面右、dy 正＝往下）
        r = p["rel"]
        pm = np.asarray(_one_component(person_mask(im).point(lambda x: 255 if x > 100 else 0), r.get("person_seed"))) > 0
        ys, xs = np.where(pm)
        top = ys.min()
        hh = r["head_h"] * H if r.get("head_h") else (ys.max() - top) / 7.5
        rows = pm[top:top + max(2, int(hh))]
        hx = np.where(rows)[1].mean()
        p["cx"] = (hx + r.get("dx", 0) * hh) / W
        p["cy"] = (top + r.get("dy", 0.5) * hh) / H
        p["diam"] = r.get("k", 1.2) * hh / H
    d = drone_sprite(p["diam"] * H, p.get("yaw", 0.0))
    out = im.convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    layer.alpha_composite(d, (round(p["cx"] * W - d.width / 2), round(p["cy"] * H - d.height / 2)))
    if p.get("behind"):
        pm = _one_component(person_mask(im).point(lambda x: 255 if x > p.get("thr", 100) else 0), p.get("person_seed")).filter(ImageFilter.GaussianBlur(0.8))
        a = np.asarray(layer.getchannel("A")).astype(float) * (1 - np.asarray(pm).astype(float) / 255)
        layer.putalpha(Image.fromarray(a.astype("uint8")))
    out = Image.alpha_composite(out, layer)
    return out.convert("RGB")


def op_neon_far(im, p):
    """街尾很遠處的一小片彩色霓虹（洋紅、青、琥珀的小亮塊＋柔光暈）。cx、cy＝中心，w、h＝範圍（對畫面高）。"""
    rnd = np.random.RandomState(p.get("seed", 3))
    cols = [(208, 64, 200), (41, 211, 245), (255, 176, 64)]
    lay = np.zeros((H, W, 3), float)
    cx, cy, w, h = p["cx"] * W, p["cy"] * H, p["w"] * H, p["h"] * H
    for _ in range(p.get("n", 26)):
        x = cx + rnd.uniform(-0.5, 0.5) * w
        y = cy + rnd.uniform(-0.5, 0.5) * h
        bw, bh = rnd.uniform(0.04, 0.10) * w, rnd.uniform(0.05, 0.16) * h
        c = np.array(cols[rnd.randint(3)], float)
        lay[max(0, int(y - bh / 2)):int(y + bh / 2) + 1, max(0, int(x - bw / 2)):int(x + bw / 2) + 1] = c
    core = Image.fromarray(lay.clip(0, 255).astype("uint8"))
    glow = np.asarray(core.filter(ImageFilter.GaussianBlur(h * 0.10))).astype(float) * 1.6
    out = np.asarray(im).astype(float) + glow * 0.9 + np.asarray(core).astype(float) * 0.9
    res = Image.fromarray(out.clip(0, 255).astype("uint8"))
    if p.get("behind_sky"):   # 只在天空（暗色連通區）上，前面的車與樓擋住它
        return Image.composite(res, im, sky_mask(im, p.get("cx", 0.5), p.get("sky_y", 0.27), p.get("sky_thr", 80)))
    return res


def find_vp_y(im, x0=0.35, x1=0.65, y0=0.35, y1=0.85):
    """街尾彩色霓虹小塊（飽和度高）的中心高度＝這張場景圖的地平線／消失點（比例）。找不到回傳 None。"""
    a = np.asarray(im).astype(float)
    sat = (a.max(axis=2) - a.min(axis=2)) / np.maximum(a.max(axis=2), 1)
    m = (sat > 0.5) & (a.max(axis=2) > 140)
    box = np.zeros_like(m)
    box[round(y0 * H):round(y1 * H), round(x0 * W):round(x1 * W)] = True
    ys, xs = np.where(m & box)
    return float(np.median(ys) / H) if len(ys) > 20 else None


def op_horizon(im, p):
    """把場景的地平線（消失點）移到指定高度：垂直分段線性變形，地平線以上壓縮、以下拉伸（Qwen 的街景消失點常在畫面高 57–62%，劇本要 45%）。
    from＝原地平線高度（比例；'auto' 依街尾霓虹的位置偵測，偵測不到用 fallback）；to＝目標高度。"""
    import cv2
    src = p.get("from", "auto")
    if src == "auto":
        src = find_vp_y(im) or p.get("fallback", 0.59)
    to = p["to"]
    yp = np.arange(H, dtype=np.float32) / H
    ys = np.where(yp <= to, yp * (src / to), src + (yp - to) * (1 - src) / (1 - to)) * H
    mapy = np.repeat(ys[:, None], W, axis=1).astype(np.float32)
    mapx = np.repeat(np.arange(W, dtype=np.float32)[None, :], H, axis=0)
    out = cv2.remap(np.asarray(im), mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out)


def op_fit_person(im, p):
    """把 Qwen 畫得太大的人物縮到劇本尺寸：用 BiRefNet 前景遮罩（最大連通區）量出人物頭頂與腳底，等比縮放並平移，讓身高＝height、腳底＝feet（皆為畫面比例）、水平中心＝cx（省略則保持中心不動）。
    crop_bottom＝腰以上的鏡頭（見程式）：只指定頭頂 top。縮小後露出的邊用邊緣像素延伸補（背景是失焦的街道，之後的 B 會重繪；鏡射會把人物頭頂鏡射進來，不用）。feet_from＝'mask'（遮罩最低點）或 'ratio'：以 feet_ratio 指定腳底在遮罩高度的比例位置（遮罩含水中倒影時用）。"""
    import cv2
    pm = _one_component(person_mask(im).point(lambda x: 255 if x > 100 else 0), p.get("person_seed"))
    ys, xs = np.where(np.asarray(pm) > 0)
    top, bot = ys.min(), ys.max()
    if p.get("crop_bottom"):   # 腰以上的鏡頭：人物下緣被畫面切掉，只量頭頂；縮放讓頭頂＝top、下緣仍貼畫面底
        s = (1 - p["top"]) * H / (H - top)
        dy = p["top"] * H - top * s
        hx = np.median(xs[ys < top + 0.2 * (bot - top)])
        dx = (p["cx"] * W - hx * s) if p.get("cx") is not None else (W - W * s) / 2
        out = cv2.warpAffine(np.asarray(im), np.float32([[s, 0, dx], [0, s, dy]]), (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(out)
    if p.get("feet_ratio"):
        bot = top + (bot - top) * p["feet_ratio"]
    elif p.get("jump", True):   # 遮罩若連腳邊的水花、水窪、倒影一起圈進來：腿的寬度很窄，往下第一次突然變寬（>2.6 倍腿寬）的那一列再往下一點＝腳底
        mk = np.asarray(pm) > 0
        w = mk.sum(axis=1).astype(float)
        lo, hi = int(top + 0.50 * (bot - top)), int(top + 0.70 * (bot - top))
        legw = np.median(w[lo:hi][w[lo:hi] > 0]) if (w[lo:hi] > 0).any() else 0
        if legw:
            idx = np.where(w[lo:bot] > 2.6 * legw)[0]
            if len(idx):
                bot = lo + idx[0] + p.get("jump_off", 22)
    s = p["height"] * H / (bot - top)
    dy = (p["feet"] - p["height"]) * H - top * s
    cxm = np.median(xs[ys < top + (bot - top) * 0.25])   # 頭部的水平中心
    dx = (p["cx"] * W - cxm * s) if p.get("cx") is not None else (W - W * s) / 2
    M = np.float32([[s, 0, dx], [0, s, dy]])
    out = cv2.warpAffine(np.asarray(im), M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out)


def op_move_balloon(im, p):
    """氣球（Qwen 常畫得太大、位置不對）：用紅色色相遮罩抠出氣球本體，原處用 cv2.inpaint 補（背景是失焦的），縮放貼到劇本指定的位置與直徑，
    再用程式畫一條細白繩（Qwen 的細繩色相不是紅、抠不到，所以一律重畫）。cx、cy＝氣球中心、diam＝直徑（對畫面高）、string＝繩子下端的高度（畫面比例，省略＝不畫繩）。
    氣球顏色維持 Qwen 的紅色（首幀由後製灰化、末幀保持紅色）。"""
    import cv2
    from scipy import ndimage as ndi
    rng = p.get("hue", {"h": [335, 25], "s": 0.28, "v": 0.2})
    m = np.asarray(hue_mask(im, [dict(rng, **({"rect": p["rect"]} if p.get("rect") else {}))])) > 0
    body = ndi.binary_opening(m, iterations=5)
    lab, n = ndi.label(body)
    if n == 0:
        return im
    k = 1 + int(np.argmax(ndi.sum(body, lab, range(1, n + 1))))
    body = ndi.binary_fill_holes(ndi.binary_closing(lab == k, iterations=6))
    ys, xs = np.where(body)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    a = np.asarray(im)
    af = a.astype(float)
    sat = (af.max(axis=2) - af.min(axis=2)) / np.maximum(af.max(axis=2), 1)
    near = ndi.binary_dilation(body, iterations=34)
    rem = ndi.binary_dilation(body, iterations=14) | (near & (sat > 0.16))     # 去除範圍：氣球本體往外 14 px，加上附近還帶紅色的像素（暗紅邊、光暈）
    if p.get("clear_string", True):   # Qwen 自己畫的細繩（白色）：從結往下一條窄帶一起補掉
        kx = int((x0 + x1) / 2)
        sd = min(H, y1 + int(p.get("clear_len", 0.45) * H))
        rem[y1 - 4:sd, max(0, kx - 14):kx + 14] = True
    hole = (ndi.binary_dilation(rem, iterations=3) * 255).astype("uint8")
    filled = cv2.inpaint(a, hole, 11, cv2.INPAINT_TELEA)
    filled = np.asarray(Image.fromarray(filled).filter(ImageFilter.GaussianBlur(2.0)))
    base = Image.fromarray(np.where(hole[..., None] > 0, filled, a).astype("uint8"))
    pad = 6
    box = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
    alpha = Image.fromarray((ndi.binary_dilation(body, iterations=2) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.0)).crop(box)
    cut = Image.fromarray(a).convert("RGBA").crop(box)
    cut.putalpha(alpha)
    D = max(x1 - x0, y1 - y0)
    f = p["diam"] * H / D
    cut = cut.resize((max(2, round(cut.width * f)), max(2, round(cut.height * f))), Image.LANCZOS)
    cxs, cys = (x0 + x1) / 2 - box[0], (y0 + y1) / 2 - box[1]
    out = base.convert("RGBA")
    px, py = round(p["cx"] * W - cxs * f), round(p["cy"] * H - cys * f)
    out.alpha_composite(cut, (px, py))
    if p.get("string"):   # 細白繩：從氣球下方的結往下垂，輕微彎曲
        S = 3
        lay = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        r = p["diam"] * H / 2
        sx, sy = p["cx"] * W, p["cy"] * H + r * 1.08
        ey = p["string"] * H
        pts = [(sx + math.sin((yy - sy) / max(1, ey - sy) * math.pi * 1.3) * p.get("sway", 5), yy) for yy in np.linspace(sy, ey, 40)]
        d.line([(x * S, y * S) for x, y in pts], fill=(236, 238, 242, 235), width=max(S, round(p.get("lw", 2) * S)))
        out = Image.alpha_composite(out, lay.resize((W, H), Image.LANCZOS))
    return out.convert("RGB")


_REF_CUT = {}


def ref_cut(ref, rect, drop_white=False):
    """從 refs 圖摳出一個物件（BiRefNet 前景遮罩 ∩ 矩形，矩形是來源圖的像素座標）：回傳裁好的 RGBA。快取。"""
    key = (ref, tuple(rect), drop_white)
    if key not in _REF_CUT:
        im = Image.open(ROOT / ref).convert("RGB")
        m = person_mask(im)
        box = tuple(rect)
        a = im.convert("RGBA").crop(box)
        al = np.asarray(m.crop(box).point(lambda x: 255 if x > 110 else 0)).astype(float)
        rgb = np.asarray(a.convert("RGB")).astype(int)
        if drop_white:   # 白色細繩等：近白色且低飽和的像素去掉
            al[(rgb.min(axis=2) > 205)] = 0
        a.putalpha(Image.fromarray(al.astype("uint8")).filter(ImageFilter.GaussianBlur(0.8)))
        _REF_CUT[key] = a
    return _REF_CUT[key].copy()


BALLOON_REF = ("refs/entities/balloon_child/back.png", (505, 30, 730, 292))     # 紅氣球本體（含結）
CHILDHAND_REF = ("refs/entities/balloon_child/back.png", (255, 395, 615, 640))   # 小孩的帽子頂與伸出的手


def op_paste_balloon(im, p):
    """氣球（程式貼上）：從 refs/entities/balloon_child/back.png 摳出的紅氣球，貼到指定位置與直徑（對畫面高），再畫一條細白繩（可被人物擋住）。
    cx、cy＝氣球中心；或 anchor='fingertip'：依 LUNA 前景遮罩最高點（舉起的指尖）定位，氣球底部在指尖上方 gap（畫面高比例）；
    grey＝true 時去飽和（frozen 的灰色氣球，亮度不變）；string＝繩子下端的高度（畫面比例）。"""
    cut = ref_cut(*BALLOON_REF)
    pm = None
    if p.get("anchor") == "fingertip" or p.get("string"):
        pm = _one_component(person_mask(im).point(lambda x: 255 if x > 100 else 0), p.get("person_seed"))
    diam = p["diam"] * H
    if p.get("anchor") == "fingertip":
        ys, xs = np.where(np.asarray(pm) > 0)
        top = ys.min()
        cx = float(np.median(xs[ys < top + 6])) + p.get("dx", 0.0) * W
        cy = top - p.get("gap", 0.012) * H - diam / 2
    else:
        cx, cy = p["cx"] * W, p["cy"] * H
    f = diam / max(cut.width, cut.height * (cut.width / cut.height))
    body_h = cut.height  # 含結的高度；直徑以寬度為準
    f = diam / cut.width
    c2 = cut.resize((max(2, round(cut.width * f)), max(2, round(cut.height * f))), Image.LANCZOS)
    if p.get("grey"):
        g = ImageOps.grayscale(c2.convert("RGB")).convert("RGBA")
        g.putalpha(c2.getchannel("A"))
        c2 = g
    out = im.convert("RGBA")
    px, py = round(cx - c2.width / 2), round(cy - diam / 2)
    if p.get("string"):
        S = 3
        lay = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        sx, sy = cx, py + c2.height - 1
        ey = p["string"] * H
        hx = None
        if p.get("string_to") == "hand" and _HAND.get("pt"):   # 繩子下端接到小孩舉起的手
            hx, ey = _HAND["pt"][0] - 3, _HAND["pt"][1] + 4
        if hx is not None:
            pts = [(sx + (hx - sx) * (yy - sy) / max(1, ey - sy) + math.sin((yy - sy) / max(1, ey - sy) * math.pi) * p.get("sway", 4), yy) for yy in np.linspace(sy, ey, 40)]
        else:
            pts = [(sx + math.sin((yy - sy) / max(1, ey - sy) * math.pi * 1.2) * p.get("sway", 4), yy) for yy in np.linspace(sy, ey, 40)]
        col = (168, 168, 168, 230) if p.get("grey") else (240, 240, 244, 235)
        d.line([(x * S, y * S) for x, y in pts], fill=col, width=max(S, round(p.get("lw", 2) * S)))
        lay = lay.resize((W, H), Image.LANCZOS)
        if pm is not None:
            a = np.asarray(lay.getchannel("A")).astype(float) * (1 - np.asarray(pm.filter(ImageFilter.GaussianBlur(0.8))).astype(float) / 255)
            lay.putalpha(Image.fromarray(a.astype("uint8")))
        out = Image.alpha_composite(out, lay)
    out.alpha_composite(c2, (px, py))
    return out.convert("RGB")


def op_paste_childhand(im, p):
    """s011 首幀：畫面下緣失焦的小孩帽子頂與伸出的手（從 refs/entities/balloon_child/back.png 摳出，去飽和＋模糊）。
    cx＝水平中心（畫面比例）、top＝帽子頂的高度（畫面比例）、width＝這塊的寬度（畫面寬比例），超出畫面下緣的部分被裁掉。"""
    cut = ref_cut(*CHILDHAND_REF, drop_white=True)
    w = p["width"] * W
    f = w / cut.width
    c2 = cut.resize((max(2, round(cut.width * f)), max(2, round(cut.height * f))), Image.LANCZOS)
    if p.get("grey", True):
        g = ImageOps.grayscale(c2.convert("RGB")).convert("RGBA")
        g.putalpha(c2.getchannel("A"))
        c2 = g
    c2 = c2.filter(ImageFilter.GaussianBlur(p.get("blur", 4)))
    out = im.convert("RGBA")
    out.alpha_composite(c2, (round(p["cx"] * W - c2.width * 0.4), round(p["top"] * H)))
    return out.convert("RGB")


_HAND = {}
_HEAD = {}


def op_paste_asset(im, p, ctx=None):
    """貼素材人物（tools/p6_make_asset.py 出的白底圖，摳圖後縮放）：file＝來源（{a}＝依 seed 輪流的素材編號，assets＝素材張數）、h＝身高（畫面高比例）、fx、fy＝腳底中心；
    grey＝true 去飽和（frozen）；flip＝水平翻轉。記下上半身最靠右的點（伸出的手）供 paste_balloon 的 string_to: hand 用。"""
    seed = ctx[2] if ctx else 1
    a = ((seed - 1) % p.get("assets", 1)) + 1
    cut = cutout_sprite(ROOT / p["file"].format(a=a, seed=seed))
    if p.get("flip"):
        cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
    f = p["h"] * H / cut.height
    cut = cut.resize((max(2, round(cut.width * f)), max(2, round(cut.height * f))), Image.LANCZOS)
    if p.get("grey"):
        g = ImageOps.grayscale(cut.convert("RGB")).convert("RGBA")
        g.putalpha(cut.getchannel("A"))
        cut = g
    if p.get("blur"):   # 景深（略失焦）
        cut = cut.filter(ImageFilter.GaussianBlur(p["blur"]))
    fx, fy = p["fx"] * W, p["fy"] * H
    out = im.convert("RGBA")
    sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse((fx - cut.width * 0.45, fy - cut.height * 0.03, fx + cut.width * 0.45, fy + cut.height * 0.03), fill=(0, 0, 0, 90))
    out = Image.alpha_composite(out, sh.filter(ImageFilter.GaussianBlur(cut.height * 0.02)))
    ox, oy = round(fx - cut.width / 2), round(fy - cut.height)
    out.alpha_composite(cut, (ox, oy))
    al = np.asarray(cut.getchannel("A")) > 128
    ys, xs = np.where(al)
    up = ys < ys.min() + 0.45 * (ys.max() - ys.min())   # 上半身裡最靠右（朝前伸出）的點＝伸出的手
    k = np.argmax(np.where(up, xs, -1))
    _HAND["pt"] = (ox + xs[k], oy + ys[k])
    return out.convert("RGB")


def op_paste_head(im, p, ctx=None):
    """s014：畫面左緣失焦的 LUNA 後腦與肩膀。取 s009 首幀的背面立繪摳圖（shots/<files>，依 seed 輪流）的上半身（frac＝取上面多少比例），
    縮到寬度 w（畫面寬比例）、頭的中心在 head_x（畫面寬比例）、下緣貼畫面底，再高斯模糊 blur px（景深）。"""
    seed = ctx[2] if ctx else 1
    f = p["files"][(seed - 1) % len(p["files"])]
    cut = Image.open(ROOT / "shots" / f).convert("RGBA")
    cut = cut.crop((0, 0, cut.width, int(cut.height * p.get("frac", 0.36))))
    s = p["w"] * W / cut.width
    cut = cut.resize((round(cut.width * s), round(cut.height * s)), Image.LANCZOS)
    if p.get("blur"):
        cut = cut.filter(ImageFilter.GaussianBlur(p["blur"]))
    out = im.convert("RGBA")
    pos = (round(p["head_x"] * W - cut.width / 2), H - cut.height)
    out.alpha_composite(cut, pos)
    hm = Image.new("L", (W, H), 0)
    hm.paste(cut.getchannel("A"), pos)
    _HEAD["mask"] = hm
    return out.convert("RGB")


# ---------------------------------------------------------------- 審查後新增的通用後製（雨點、眼睛、看板、整合）
def _person_bin(im, seed=None, thr=100):
    return _one_component(person_mask(im).point(lambda x: 255 if x > thr else 0), seed)


def op_remove_rain(im, p):
    """去掉雨絲與水滴（B 之前清 A、或清場景圖）：白頂帽（亮的細小結構＝比 size 小的亮色筆畫）超過 thr 的像素用 cv2.inpaint 補掉。
    person＝true 時用 BiRefNet 遮罩保護人物（seed＝person_seed）；ymax＝只清畫面高這個比例以上的部分（s010 末幀腳邊的水花要留）。"""
    import cv2
    a = np.asarray(im)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.get("size", 15), p.get("size", 15)))
    th = g.astype(int) - cv2.morphologyEx(g, cv2.MORPH_OPEN, k).astype(int)
    m = (th > p.get("thr", 22)).astype("uint8")
    if p.get("ymax"):
        m[round(p["ymax"] * H):] = 0
    if p.get("person", True):
        pm = np.asarray(_person_bin(im, p.get("person_seed"))) > 0
        pm = cv2.dilate(pm.astype("uint8"), np.ones((13, 13), np.uint8)) > 0
        m[pm] = 0
    m = cv2.dilate(m, np.ones((5, 5), np.uint8)) * 255
    if p.get("person", True):
        m[pm] = 0
    return Image.fromarray(cv2.inpaint(a, m, 4, cv2.INPAINT_TELEA))


def op_frozen_rain(im, p, ctx=None):
    """停在半空的灰色小圓點雨滴（劇本：frozen 區的雨）：整個畫面散布，偏向近處（下方）、近大遠小；人物遮罩內不畫。
    n＝數量、rmin／rmax＝半徑 px、seed 依 ctx 的 seed 變化。"""
    seed = (ctx[2] if ctx else 0) + p.get("seed", 5)
    rnd = np.random.RandomState(seed)
    S = 2
    lay = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for _ in range(p.get("n", 320)):
        x, y = rnd.uniform(0, W), (rnd.uniform(0, 1) ** 0.75) * H
        r = (p.get("rmin", 1.2) + (p.get("rmax", 3.2) - p.get("rmin", 1.2)) * (0.3 + 0.7 * y / H) * rnd.uniform(0.6, 1.1)) * S
        al = int(rnd.uniform(120, 200))
        d.ellipse((x * S - r, y * S - r, x * S + r, y * S + r), fill=(196, 202, 210, al))
        d.ellipse((x * S - r * 0.45, y * S - r * 0.6, x * S - r * 0.05, y * S - r * 0.2), fill=(240, 244, 248, min(255, al + 40)))
    lay = lay.resize((W, H), Image.LANCZOS)
    if p.get("person", True):
        pm = _person_bin(im, p.get("person_seed")).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(2))
        a = np.asarray(lay.getchannel("A")).astype(float) * (1 - np.asarray(pm).astype(float) / 255)
        lay.putalpha(Image.fromarray(a.astype("uint8")))
    return Image.alpha_composite(im.convert("RGBA"), lay).convert("RGB")


def op_fix_eyes(im, p):
    """虹膜顏色修正（style bible：深紫灰）：B 偶爾把大張的眼睛畫成棕或紅棕。在人物頭部範圍內找「橘棕／紅棕、飽和、偏暗」的像素
    （膚色亮、頭髮紫、嘴唇在更下面，所以用亮度＋位置分開），色相改成紫灰、保留亮度起伏。
    roi_h＝頭部高度（人物遮罩高度的比例，全身約 0.17、腰以上約 0.5），band＝眼睛所在的範圍（頭部高度比例，從頭頂算）。"""
    import cv2
    pm = np.asarray(_person_bin(im, p.get("person_seed"))) > 0
    ys, xs = np.where(pm)
    top, bot = ys.min(), ys.max()
    hh = (bot - top) * p.get("roi_h", 0.17)
    r0, r1 = int(top + p.get("band", [0.3, 0.75])[0] * hh), int(top + p.get("band", [0.3, 0.75])[1] * hh)
    hsv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HSV).astype(float)
    h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
    sel = (((h <= 55) | (h >= 330)) & (s > 0.26) & (v < p.get("vmax", 0.72)) & (v > 0.10))
    box = np.zeros_like(sel)
    box[r0:r1] = True
    sel &= box & pm
    from scipy import ndimage as ndi
    lab, n = ndi.label(ndi.binary_opening(sel, iterations=1) if p.get("open", False) else sel)
    if n == 0:
        return im
    keep = np.zeros_like(sel)
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() >= p.get("min_px", 3):
            keep |= c
    keep = ndi.binary_dilation(keep, iterations=1)
    sel2 = keep & (((h <= 62) | (h >= 310)) & (s > 0.18) & (v < 0.78))
    m = Image.fromarray((sel2 * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.7))
    tgt = np.array(p.get("color", [88, 76, 104]), float)   # 深紫灰
    base = np.asarray(im).astype(float)
    lum = (0.3 * base[..., 0] + 0.59 * base[..., 1] + 0.11 * base[..., 2]) / 255
    new = tgt[None, None, :] * (0.55 + 0.9 * lum)[..., None] / 1.0
    new = np.clip(new, 0, 255)
    al = (np.asarray(m).astype(float) / 255)[..., None]
    return Image.fromarray((base * (1 - al) + new * al).astype("uint8"))


_BB_CACHE = {}


def op_billboards(im, p, ctx=None):
    """看板（劇本：看板仍亮但沒有顏色）：Qwen 把「空白看板」畫成純白方塊。找出近白的大矩形，換成灰階的畫面內容
    （refs/entities/monitor_wall/city_*.png 的城市畫面去飽和、壓到亮度 lo–hi、略模糊；沒有文字），亮度高於周圍但不是純白。
    thr＝近白門檻、min_area＝最小面積 px、lo／hi＝內容亮度範圍、blur。依 seed 與面板序號輪流選內容。"""
    import cv2
    from scipy import ndimage as ndi
    seed = ctx[2] if ctx else 0
    files = sorted((ROOT / "refs/entities/monitor_wall").glob("city_*.png"))
    a = np.asarray(im)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    m = (g > p.get("thr", 236)).astype("uint8")
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lab, n = ndi.label(m)
    out = a.astype(float)
    done = 0
    for i in range(1, n + 1):
        comp = (lab == i).astype("uint8")
        if comp.sum() < p.get("min_area", 1500):
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        per = cv2.arcLength(hull, True)
        ap = cv2.approxPolyDP(hull, 0.03 * per, True)
        quad = ap.reshape(-1, 2).astype(np.float32) if len(ap) == 4 else cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        s_ = quad.sum(axis=1)
        d_ = np.diff(quad, axis=1).ravel()
        order = [np.argmin(s_), np.argmin(d_), np.argmax(s_), np.argmax(d_)]   # 左上、右上、右下、左下
        quad = quad[order]
        f = files[(seed + done) % len(files)]
        if f not in _BB_CACHE:
            c0 = np.asarray(ImageOps.grayscale(Image.open(f).convert("RGB"))).astype(float)
            lo_, hi_ = np.percentile(c0, 2), np.percentile(c0, 98)
            c0 = np.clip((c0 - lo_) / max(1, hi_ - lo_), 0, 1)
            _BB_CACHE[f] = (c0 * (p.get("hi", 214) - p.get("lo", 120)) + p.get("lo", 120))
        c0 = _BB_CACHE[f]
        sh, sw = c0.shape
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]), quad)
        if not np.isfinite(M).all():
            continue
        warped = np.nan_to_num(cv2.warpPerspective(c0.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), nan=150.0, posinf=214.0, neginf=120.0)
        warped = cv2.GaussianBlur(warped, (0, 0), p.get("blur", 1.6))
        mm = cv2.erode(comp, np.ones((3, 3), np.uint8)).astype(float)
        mm = cv2.GaussianBlur(mm, (0, 0), 0.8)[..., None]
        out = out * (1 - mm) + warped[..., None].repeat(3, axis=2) * mm
        done += 1
    return Image.fromarray(out.clip(0, 255).astype("uint8"))


def op_shift_person(im, p):
    """把人物整體往下移到指定位置（s011 末幀：舉高的指尖在畫面最上緣，氣球放不進畫面）：用 BiRefNet 遮罩最高點（指尖）定位，
    往下移到 tip_y（畫面比例）；上面露出的邊用邊緣像素延伸（之後的 B 會重繪）。"""
    import cv2
    pm = _person_bin(im, p.get("person_seed"))
    ys, xs = np.where(np.asarray(pm) > 0)
    top = ys.min()
    dy = p["tip_y"] * H - top
    if dy <= 0:
        return im
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    return Image.fromarray(cv2.warpAffine(np.asarray(im), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def lora_pass(im, tags, seed, denoise, canny=0.6, canny_end=0.6, neg_add="", key="h"):
    """對整幅（1344×768）做一次 Illustrious＋LoRA img2img（canny 固定構圖）：低 denoise 時只整合光影與線條，不改位置。"""
    import os
    pos = L.BASE + L.prompt_outfit(tags) + tags + L.QUALITY
    tmp = ROOT / "shots" / f".p6_tmp_h_{os.getpid()}.png"
    im.convert("RGB").save(tmp)
    inp = K.put(tmp, f"p6_h_{os.getpid()}_{key}.png")
    tmp.unlink()
    wf = L.workflow(K.LORA, 0.9, pos, W, H, seed, 28, 6.0)
    if neg_add:
        wf["5"]["inputs"]["text"] = L.NEG + ", " + neg_add
    wf["30"] = {"class_type": "LoadImage", "inputs": {"image": inp}}
    wf["31"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["30", 0], "vae": ["1", 2]}}
    wf["7"]["inputs"]["latent_image"] = ["31", 0]
    wf["7"]["inputs"]["denoise"] = denoise
    del wf["6"]
    wf["20"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": L.CONTROLNET}}
    wf["21"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["20", 0], "type": "canny/lineart/anime_lineart/mlsd"}}
    wf["22"] = {"class_type": "Canny", "inputs": {"image": ["30", 0], "low_threshold": 0.3, "high_threshold": 0.6}}
    wf["23"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "control_net": ["21", 0], "image": ["22", 0],
                                                                     "strength": canny, "start_percent": 0.0, "end_percent": canny_end, "vae": ["1", 2]}}
    wf["7"]["inputs"]["positive"], wf["7"]["inputs"]["negative"] = ["23", 0], ["23", 1]
    data = K.run(wf, "9")
    import io
    return Image.open(io.BytesIO(data)).convert("RGB")


def op_harmonize(im, p, ctx=None):
    """貼上的人物（立繪、素材）看起來像剪貼：整幅再做一次低 denoise（約 0.3–0.4）的 B，整合光影與線條；canny 固定，位置不動。"""
    seed = ctx[2] if ctx else 1
    return lora_pass(im, p["tags"], p.get("seed", seed), p.get("denoise", 0.35), p.get("canny", 0.7), p.get("canny_end", 0.8), p.get("neg_add", ""), "h")


def op_desat_hue(im, p):
    """某個色相範圍去飽和（frozen 的水花被 B 染到青色、但在人物遮罩內不會被灰化時用）：h＝[h0,h1] 度，s＝最低飽和度，rect＝限制範圍（畫面比例）。"""
    import cv2
    a = np.asarray(im)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(float)
    h, s = hsv[..., 0] * 2, hsv[..., 1] / 255
    m = (h >= p["h"][0]) & (h <= p["h"][1]) & (s > p.get("s", 0.12))
    if p.get("rect"):
        x0, y0, x1, y1 = p["rect"]
        b = np.zeros_like(m); b[round(y0 * H):round(y1 * H), round(x0 * W):round(x1 * W)] = True
        m &= b
    mm = Image.fromarray((m * 255).astype("uint8")).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.0))
    return Image.composite(ImageOps.grayscale(im).convert("RGB"), im, mm)


OPS = dict(desat_hue=op_desat_hue, remove_rain=op_remove_rain, frozen_rain=op_frozen_rain, fix_eyes=op_fix_eyes, billboards=op_billboards, shift_person=op_shift_person, harmonize=op_harmonize, paste_head=op_paste_head, paste_asset=op_paste_asset, paste_balloon=op_paste_balloon, paste_childhand=op_paste_childhand, move_balloon=op_move_balloon, fit_person=op_fit_person, horizon=op_horizon, drone=op_drone, neon_far=op_neon_far, grey_keep=op_grey_keep, skyline_from=op_skyline_from, roof=op_roof, floor_stretch=op_floor_stretch, band_grey=op_band_grey, tower=op_tower, grey=op_grey, glow=op_glow, hatch=op_hatch, paste_drone=op_paste_drone, platform_dot=op_platform_dot, shift=op_shift, zoom=op_zoom)


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
        if (stage in ("all", "A") and (force or not fa.exists()) and not v.get("scene_a")) or (v.get("scene") == "first" and (not fa.exists() or stage in ("post", "cut", "BC"))):
            if v.get("scene") == "first":
                ss = v.get("srcseed", (seed - 1) % 6 + 1)   # mid 的 seed 7–12 沿用首幀 seed 1–6 的場景
                shutil.copy(cdir(shot) / f"first_Cpre_s{ss}.png", fa)
                log(shot, dict(step="A", frame=frame, file=fa.name, seed=seed, refs=[f"first_Cpre_s{ss}.png"],
                               prompt="(沿用同 seed 首幀的場景，不經 Qwen)", seconds=0))
            else:
                step_A(shot, frame, seed, v, macros, None)
        if v.get("scene_a") and (force or not fa.exists() or stage in ("post", "cut", "BC")):   # 場景沿用另一幀（同鏡或他鏡）的 A 稿，首尾幀同一個場景
            ssd = (v.get("scene_map") or {}).get(seed, seed)   # scene_map：這個 seed 的場景借用哪個 seed 的 A 稿（補出新候選時用）
            src = ROOT / "shots" / v["scene_a"].format(seed=ssd) if "/" in v["scene_a"] else cdir(shot) / v["scene_a"].format(seed=ssd)
            shutil.copy(src, fa)
            log(shot, dict(step="A", frame=frame, file=fa.name, seed=seed, refs=[str(src.relative_to(ROOT))], prompt="(沿用 %s 的場景，不經 Qwen)" % src.name, seconds=0))
        if v.get("cut_from"):   # 立繪沿用另一幀的摳圖（同一個人物，已含翻轉），首尾幀的外觀才一致
            cp = cdir(shot) / f"{frame}_cut_s{seed}.png"
            if force or not cp.exists() or stage in ("cut", "BC"):
                src = ROOT / "shots" / v["cut_from"].format(seed=seed) if "/" in v["cut_from"] else cdir(shot) / v["cut_from"].format(seed=seed)
                shutil.copy(src, cp)
        elif stage in ("all", "B", "BC") and (force or not (cdir(shot) / f"{frame}_cut_s{seed}.png").exists()):
            step_sprite(shot, frame, seed, v)
        if stage == "cut" and not v.get("cut_from"):   # 只從已存的立繪重新摳圖（改摳圖演算法或 flip 清單後用），再做後製
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
