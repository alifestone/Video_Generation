#!/usr/bin/env python3
"""P4 Intro 試作片 s002–s004 後製（frame 67–230，1920x1080 24fps）。
~/creative/ComfyUI/comfyui/bin/python tools/p4_intro_post.py [--cache DIR] [--out pilot/intro_pilot.mp4] [--no-esrgan]
流程：
 s002  keyframe 逐拍亮螢幕（screens.json bbox）＋灰白室內光＋主螢幕波形抖動 → 視差推近（深度圖）→ ESRGAN 1344x756→1080p
 s003  H3 turbo 片段（24fps）重定時（ECHO 入畫＝bar2 第2拍、停穩＝第3拍）→ ESRGAN → ECHO 光帶 glow_pulse＋@countdown_ring
 s004  Wan FLF（pilot/wan22_lightx2v_flf/s004.mp4）16fps → RIFE48（pilot/post_tests/rife_s004_flf_48.mp4，tools/p4_post.py rife）取幀（睜眼起點對準 8.30 s）→ ESRGAN → 玻璃弧形反光＋瞳孔青環
時間一律來自 analysis/timeline.json；frame = round(t*24)。事件預定幀與切點寫在 pilot/post_checks/plan.json。
"""
import argparse, json, subprocess, sys, os, time
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OW, OH = 1920, 1080
CYAN = np.array([0x29, 0xd3, 0xf5], np.float32) / 255
GRAYLIGHT = np.array([0xb0, 0xb4, 0xba], np.float32) / 255
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=str(ROOT / "pilot/post_checks/_cache"))
ap.add_argument("--out", default=str(ROOT / "pilot/intro_pilot.mp4"))
ap.add_argument("--no-esrgan", action="store_true")
A = ap.parse_args()
CACHE = Path(A.cache); CACHE.mkdir(parents=True, exist_ok=True)
CHK = ROOT / "pilot/post_checks"; CHK.mkdir(parents=True, exist_ok=True)

# ───────── 時間（timeline.json）─────────
TL = json.load(open(ROOT / "analysis/timeline.json"))
beats = {(b["bar"], b["beat_in_bar"]): b["frame"] for b in TL["beats"]}
bars = {b["bar"]: (b["frame_start"], b["frame_end"]) for b in TL["bars"]}
S2 = bars[1]; S3 = bars[2]; S4 = bars[3]            # (start, end) end 不含
F0, F1 = S2[0], S4[1]
VOCAL_QI_T = 8.30
F_QI = round(VOCAL_QI_T * 24)                       # 199
SCREEN_BEATS = [beats[(1, 1)], beats[(1, 2)], beats[(1, 3)], beats[(1, 4)]]   # 67 81 94 108
F_ECHO_IN, F_ECHO_STOP = beats[(2, 2)], beats[(2, 3)]                         # 135 148
PLAN = dict(frames=[F0, F1], s002=S2, s003=S3, s004=S4, screen_beats=SCREEN_BEATS, echo_in=F_ECHO_IN,
            echo_stop_ring=F_ECHO_STOP, eye_open=F_QI)

# ───────── ESRGAN ─────────
_model = None
def esrgan(rgb):  # uint8 HxWx3 → 1920x1080 uint8
    global _model
    if A.no_esrgan:
        return cv2.resize(rgb, (OW, OH), interpolation=cv2.INTER_LANCZOS4)
    if _model is None:
        from spandrel import ModelLoader
        _model = ModelLoader().load_from_file(str(Path.home() / "creative/ComfyUI/models/upscale_models/RealESRGAN_x4plus_anime_6B.pth")).model.cuda().half().eval()
    x = torch.from_numpy(rgb.copy()).cuda().permute(2, 0, 1)[None].half() / 255
    with torch.no_grad():
        y = _model(x)
    y = (y.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255 + .5).astype(np.uint8)
    return cv2.resize(y, (OW, OH), interpolation=cv2.INTER_AREA)

def decode(path, W, H):
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(b, np.uint8).reshape(-1, H, W, 3)

def screen_blend(base, add):  # float 0-1
    return 1 - (1 - base) * (1 - np.clip(add, 0, 1))

def blur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)

def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

# ═════════ s002 ═════════
def build_s002():
    kf = cv2.cvtColor(cv2.imread(str(ROOT / "shots/s002/keyframe.png")), cv2.COLOR_BGR2RGB)   # 1344x768
    H0, W0 = kf.shape[:2]
    sc = json.load(open(ROOT / "shots/s002/screens.json"))["screens"]
    # 7 張 @city（refs/entities/monitor_wall/city_N_*.png；彩色＋灰階版另存 shots/s002/screens/）
    outd = ROOT / "shots/s002/screens"; outd.mkdir(exist_ok=True)
    cities = []
    for i in range(1, 8):
        p = sorted((ROOT / "refs/entities/monitor_wall").glob(f"city_{i}_*.png"))[0]
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]; m = int(0.045 * h); im = im[m:h - m, int(0.03 * w):w - int(0.03 * w)]   # 去掉黑邊
        g = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(outd / f"city_{i}_color.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(outd / f"city_{i}_gray.png"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
        cities.append((im, g))
    groups = {}   # (row,col) → 亮起幀
    for r in range(3):
        for c in range(7):
            if (r, c) == (1, 3): continue
            d = abs(c - 3)                          # 0 正中欄、1..3
            if d == 0 and r == 1: continue
            groups[(r, c)] = SCREEN_BEATS[d]
    # 艙遮擋：畫面中央下層螢幕被艙擋住的部分（keyframe 上非黑的像素）
    lum = cv2.cvtColor(kf, cv2.COLOR_RGB2GRAY)
    podbox = np.zeros((H0, W0), bool); podbox[455:680, 635:765] = True
    occl = podbox & (lum > 38)
    occl = cv2.dilate(occl.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # 預先把各格的城市內容縮放
    tiles = {}
    for (r, c), _ in groups.items():
        x0, y0, x1, y1 = sc[f"{r},{c}"]; x0 += 3; y0 += 3; x1 -= 3; y1 -= 3
        ci = (c + 3 * r) % 7; col, gray = cities[ci]
        tiles[(r, c)] = (x0, y0, x1, y1, cv2.resize(col, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA),
                         cv2.resize(gray, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA))
    mx0, my0, mx1, my1 = sc["1,3"]
    # 視差
    depth = cv2.imread(str(ROOT / "pilot/post_tests/s002_depth.png"), 0)
    depth = cv2.resize(depth, (W0, H0), interpolation=cv2.INTER_CUBIC)
    CY = 6; CH = 756                                 # 1344x768 → 16:9 裁成 1344x756（上下各 6）
    d = torch.from_numpy(depth[CY:CY + CH].copy()).float().cuda()[None, None] / 255
    lo, hi = torch.quantile(d, 0.01), torch.quantile(d, 0.99); d = ((d - lo) / (hi - lo)).clamp(0, 1)
    k = 9 * 2 + 1; d = F.avg_pool2d(F.pad(d, (k // 2,) * 4, mode="replicate"), k, 1)
    zoom, spread = 0.05, 0.6
    mean_d = float(d.mean()); dz = spread * zoom; z_far = zoom - dz * mean_d; z_near = z_far + dz
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, CH, device="cuda"), torch.linspace(-1, 1, W0, device="cuda"), indexing="ij")
    base = torch.stack([xx, yy], -1)[None]
    N = S2[1] - S2[0]
    rng = np.random.RandomState(7)
    jit = rng.uniform(-1, 1, N)
    frames = []; info = []
    for n in range(N):
        f = S2[0] + n
        im = kf.astype(np.float32) / 255
        lit = np.zeros((H0, W0), np.float32); screens_img = im.copy(); nlit = 0
        for (r, c), t0 in groups.items():
            kk = f - t0
            if kk < 0: continue
            x0, y0, x1, y1, col, gray = tiles[(r, c)]
            w = x1 - x0
            if kk < 3: p = 0.0
            elif kk < 9: p = (kk - 3 + 1) / 6
            else: p = 1.0
            cx = int(round(p * w))
            tile = col.copy()
            if cx > 0: tile[:, :cx] = gray[:, :cx]
            if 0 < cx < w:    # 掃描邊界 2px 柔邊
                for dx in (-1, 0):
                    xx_ = cx + dx
                    if 0 <= xx_ < w: tile[:, xx_] = (0.5 * gray[:, xx_].astype(np.float32) + 0.5 * col[:, xx_]).astype(np.uint8)
            t = tile.astype(np.float32) / 255 * 0.92
            reg = np.s_[y0:y1, x0:x1]
            keep = ~occl[reg]
            screens_img[reg] = np.where(keep[..., None], t, screens_img[reg])
            ramp = min(1.0, (kk + 1) / 3)
            lit[reg] += ramp * keep; nlit += ramp
        # 主螢幕：@signal_wave 微幅上下抖動（≤ 螢幕高 3%）
        if True:
            sh = int(round(jit[n] * 2)); ins = 3
            reg = screens_img[my0 + ins:my1 - ins, mx0 + ins:mx1 - ins]
            screens_img[my0 + ins:my1 - ins, mx0 + ins:mx1 - ins] = np.roll(reg, sh, axis=0)
        lit[my0:my1, mx0:mx1] += 0.5
        # 室內光：先對背景做 screen blend，再貼螢幕
        light = 0.50 * blur(lit, 38) + 0.075 * (nlit / 20.0)
        # 艙底座/地面略受光；螢幕正前方受光較多，離螢幕越遠越弱
        yv = np.linspace(0, 1, H0, dtype=np.float32)[:, None]
        light = light * (0.55 + 0.45 * yv)
        bg = screen_blend(im, light[..., None] * GRAYLIGHT[None, None])
        scr_mask = np.zeros((H0, W0), np.float32)
        for (r, c), t0 in groups.items():
            if f >= t0:
                x0, y0, x1, y1, _, _ = tiles[(r, c)]; scr_mask[y0:y1, x0:x1] = 1
        scr_mask[occl] = 0
        scr_mask[my0 + 3:my1 - 3, mx0 + 3:mx1 - 3] = 1
        out = bg * (1 - scr_mask[..., None]) + screens_img * scr_mask[..., None]
        # 灰階螢幕的外暈
        out = screen_blend(out, 0.10 * blur(scr_mask * (lit > 0.5), 6)[..., None] * GRAYLIGHT[None, None])
        out = out[CY:CY + CH]
        # 視差推近
        t = n / (N - 1); e = t * t * (3 - 2 * t)
        s = 1 + e * (z_far + (z_near - z_far) * d[0, 0])
        img = torch.from_numpy(out.copy()).permute(2, 0, 1)[None].float().cuda()
        w_ = F.grid_sample(img, base / s[None, :, :, None], mode="bicubic", padding_mode="border", align_corners=True).clamp(0, 1)
        rgb = (w_[0].permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
        frames.append(esrgan(rgb))
    return frames

# ═════════ ring / glow helpers (1080p) ═════════
def ring_mask(cx, cy, R=270, wd=36):
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    r = np.hypot(xx - cx, yy - cy); th = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 360
    rm = np.clip(wd / 2 + 0.5 - np.abs(r - (R - wd / 2)), 0, 1)
    am = np.zeros_like(r)
    for c in (60, 180, 300):
        dd = np.abs((th - c + 180) % 360 - 180)
        arc_half = 55
        # 角度邊緣用弧長換算成像素寬度做反鋸齒
        am = np.maximum(am, np.clip((arc_half - dd) * np.radians(1) * (R - wd / 2) + 0.5, 0, 1))
    core = np.clip(1 - np.abs(r - (R - wd / 2)) / (wd / 2), 0, 1)
    return rm * am, core * am

def band_mask(img):  # 1080p uint8 → ECHO 光帶（青色飽和像素，限 ECHO 區域）
    im = img.astype(np.int16); R, G, B = im[..., 0], im[..., 1], im[..., 2]
    m = ((R < 120) & (G > 180) & (B > 215)).astype(np.uint8)
    m[:, :1300] = 0; m[:60] = 0; m[520:] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = np.zeros(m.shape, np.float32)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a > 80 and w > 1.5 * h: out[lab == i] = 1
    return out

def glow_curve(k):   # k = 距 beat 3 的幀數
    return {0: 0.75, 1: 1.0, 2: 0.85, 3: 0.6, 4: 0.4, 5: 0.25}.get(k, 0.12 if k > 5 else 0.0)

# ═════════ s003 ═════════
EYE3 = (656, 172)       # 720p 影格內 LUNA 雙眼中點（人工標註：左眼 (630,176)、右眼 (682,168)）
def src_idx_s003(m):
    """output index m（0 起）→ H3 來源幀（浮點）。來源：右緣入畫約 13.5、位置停穩約 55、轉正完成（光帶置中）約 81。
    規格要求第2拍入畫、第3拍停穩轉正，只隔 13 幀（0.54 s），來源需 67.5 幀，所以這一段重定時（滑入約 6 倍、轉向約 4 倍）。"""
    mi = F_ECHO_IN - S3[0]; ms = F_ECHO_STOP - S3[0]; mp = mi + 7      # mp：位置停穩
    if m <= mi: return max(0.0, 13.5 - (mi - m))
    if m <= mp: return 13.5 + (m - mi) * (55 - 13.5) / (mp - mi)
    if m <= ms: return 55 + (m - mp) * (81 - 55) / (ms - mp)
    return 81 + (m - ms)

def build_s003():
    src = decode(ROOT / "pilot/h3_turbo/s003.mp4", 1280, 720)
    N = S3[1] - S3[0]
    cx, cy = EYE3[0] * 1.5, EYE3[1] * 1.5
    rm, core = ring_mask(cx, cy)
    glow = blur(rm, 14) * 1.0 + blur(rm, 40) * 0.5
    cache = {}
    frames = []; info = []
    for m in range(N):
        f = S3[0] + m
        si = min(int(round(src_idx_s003(m))), len(src) - 1)
        if si not in cache: cache[si] = esrgan(src[si])
        img = cache[si].astype(np.float32) / 255
        k = f - F_ECHO_STOP
        if k >= 0:
            # ECHO 光帶 glow_pulse
            bm = band_mask(cache[si])
            I = glow_curve(k)
            if bm.sum() > 0 and I > 0:
                g = blur(bm, 5) * 0.9 + blur(bm, 16) * 0.8 + blur(bm, 40) * 0.5
                img = screen_blend(img, I * g[..., None] * CYAN)
                img = img * (1 - 0.5 * I * bm[..., None]) + 0.5 * I * bm[..., None] * np.array([0.85, 1, 1], np.float32)
            # @countdown_ring：4 幀由無到全亮
            a = [0.25, 0.55, 0.8, 1.0][k] if k < 4 else 1.0
            ringcol = CYAN[None, None] * (1 - 0.45 * core[..., None]) + np.array([0.82, 0.97, 1.0], np.float32)[None, None] * 0.45 * core[..., None]
            img = screen_blend(img, a * 0.55 * glow[..., None] * CYAN)
            img = img * (1 - a * rm[..., None]) + a * rm[..., None] * ringcol
        frames.append((np.clip(img, 0, 1) * 255 + .5).astype(np.uint8))
        info.append(si)
    return frames, info

# ═════════ s004 ═════════
ONSET48 = 34           # rife_s004_48 中眼瞼開始動的幀（相鄰幀差尖峰，見 pilot/notes.md／報告）
def src_idx_s004(n):
    n0 = F_QI - S4[0]
    if n <= n0: return n * ONSET48 / n0             # 睜眼前的前段慢放（0.71 倍）以對準 8.30 s
    return ONSET48 + 2 * (n - n0)                     # 之後 48fps 隔幀＝24fps 原速

IRIS0 = [(340, 395), (955, 390)]    # 720p 睜眼後瞳孔中心（FLF 版，人工標註；逐幀再以暗色瞳孔重新偵測）
def iris_mask(img, c, rad):
    """img 1080p uint8；c 1080p 中心。灰紫虹膜（B≥R−10、亮度 60–215）＋黑瞳孔，限制在半徑 rad 內；排除膚色與眼白。"""
    x0, y0 = int(c[0] - rad), int(c[1] - rad); x1, y1 = int(c[0] + rad), int(c[1] + rad)
    sub = img[y0:y1, x0:x1].astype(np.int16); R, G, B = sub[..., 0], sub[..., 1], sub[..., 2]
    lum = (R + G + B) // 3
    m = ((((B - R) > -10) & (lum > 60) & (lum < 215)) | (lum < 55)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    yy, xx = np.mgrid[y0:y1, x0:x1]; m &= ((np.hypot(xx - c[0], yy - c[1]) < rad).astype(np.uint8))
    full = np.zeros((OH, OW), np.float32); full[y0:y1, x0:x1] = m
    return full, m

def pupil_center(img, c, r=75):
    x0, y0 = int(c[0] - r), int(c[1] - r)
    sub = img[y0:int(c[1] + r), x0:int(c[0] + r)].astype(np.int16).mean(-1)
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    m = (sub < 60) & (np.hypot(xx + x0 - c[0], yy + y0 - c[1]) < r)
    if m.sum() < 400: return None
    return (xx[m].mean() + x0, yy[m].mean() + y0)

def build_s004():
    src = decode(ROOT / "pilot/post_tests/rife_s004_flf_48.mp4", 1280, 720)
    N = S4[1] - S4[0]
    cache = {}; frames = []; info = []
    # 玻璃弧形反光（@countdown_ring 的倒影）：大圓的一段
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    ccx, ccy, RR = 960, 2150, 1330
    r = np.hypot(xx - ccx, yy - ccy); th = np.degrees(np.arctan2(-(yy - ccy), xx - ccx))   # 90°＝正上
    arc = np.clip(10 - np.abs(r - RR), 0, 1) * np.clip((30 - np.abs(th - 90)) / 10, 0, 1)
    arc_g = blur(arc, 3) + 0.5 * blur(arc, 12)
    ring_c = {}
    pupil_ref = [None, None]
    for n in range(N):
        f = S4[0] + n
        si = min(int(round(src_idx_s004(n))), len(src) - 1)
        if si not in cache: cache[si] = esrgan(src[si])
        base = cache[si]; img = base.astype(np.float32) / 255
        # 1. 玻璃反光：整鏡存在，睜眼後略淡；輕微閃爍
        aa = (0.34 if f < F_QI + 8 else 0.24)
        img = screen_blend(img, aa * arc_g[..., None] * CYAN * np.array([0.75, 0.95, 1.0], np.float32))
        # 2. 瞳孔倒影：睜開完成（來源 48fps 幀 ≥ 60）後淡入
        if si >= 60:
            a = min(1.0, (si - 60) / 8 + 0.25)
            for ei, (ex, ey) in enumerate(IRIS0):
                c = pupil_ref[ei] if pupil_ref[ei] is not None else (ex * 1.5, ey * 1.5)
                for _ in range(2):
                    pc = pupil_center(base, c)
                    if pc is not None: c = pc
                pupil_ref[ei] = c if pupil_ref[ei] is None else (0.5 * pupil_ref[ei][0] + 0.5 * c[0], 0.5 * pupil_ref[ei][1] + 0.5 * c[1])
                pc = pupil_ref[ei]
                mk, _ = iris_mask(base, pc, 125)
                rr = np.hypot(xx - pc[0], yy - pc[1])
                rg = np.clip(4.5 - np.abs(rr - 62), 0, 1)                    # 圓環（半徑 62、線寬約 9 px，貼在瞳孔外緣的虹膜上）
                clip = cv2.GaussianBlur(mk, (0, 0), 1.2)
                rg = rg * clip
                gl = blur(rg, 5) * clip
                img = img * (1 - a * 0.9 * rg[..., None]) + a * 0.9 * rg[..., None] * CYAN
                img = screen_blend(img, a * 0.6 * gl[..., None] * CYAN)
        frames.append((np.clip(img, 0, 1) * 255 + .5).astype(np.uint8))
        info.append(si)
    return frames, info

# ═════════ main ═════════
def main():
    t0 = time.time()
    f2 = build_s002(); print("s002", len(f2), round(time.time() - t0), "s", flush=True)
    f3, i3 = build_s003(); print("s003", len(f3), round(time.time() - t0), "s", flush=True)
    f4, i4 = build_s004(); print("s004", len(f4), round(time.time() - t0), "s", flush=True)
    frames = f2 + f3 + f4
    assert len(frames) == F1 - F0, (len(frames), F1 - F0)
    PLAN["src_idx_s003"] = i3; PLAN["src_idx_s004"] = i4
    json.dump(PLAN, open(CHK / "plan.json", "w"), indent=1)
    out = Path(A.out); tmpv = out.with_suffix(".video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", "24", "-i", "-",
                            "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", "-r", "24", str(tmpv)], stdin=subprocess.PIPE)
    for fr in frames: enc.stdin.write(fr.tobytes())
    enc.stdin.close(); enc.wait()
    dur = (F1 - F0) / 24
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmpv), "-ss", f"{F0 / 24:.6f}", "-t", f"{dur:.6f}", "-i", str(ROOT / "audio/song.wav"),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
    tmpv.unlink()
    print("done", out, round(time.time() - t0), "s")

if __name__ == "__main__":
    main()
