#!/usr/bin/env python3
"""P4 影片生成試作：比較 Wan2.2 I2V（lightx2v / 完整步數 / 首尾幀）與 MiniMax H3 FL2VA。

~/creative/ComfyUI/comfyui/bin/python tools/p4_video.py --combo wan22_lightx2v --shot s004 [--res 1280x720] [--seed N]

combo: anisora | anisora_lightx2v | wan22_lightx2v | wan22_full | wan22_lightx2v_flf | h3_turbo | h3_full
輸出：pilot/<combo>/<shot>.mp4（H.264 yuv420p、無音訊）、<shot>_sheet.png（8 格）、並把時間／RAM／VRAM 峰值與參數寫進 pilot/metrics.json。
ComfyUI 需已在 127.0.0.1:8188 執行。prompt／negative／seed 取自 pilot/prompts.yaml。
"""
import argparse, base64, json, os, shutil, socket, struct, subprocess, sys, threading, time, urllib.request, urllib.parse, uuid
from pathlib import Path

import psutil, yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
HOST = "http://127.0.0.1:8188"
COMFY = Path.home() / "creative/ComfyUI"
OUT_W, OUT_H = 1280, 720

WAN = dict(
    high="Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf", low="Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf",
    lora_high="wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
    lora_low="wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
    clip="umt5_xxl_fp8_e4m3fn_scaled.safetensors", vae="wan_2.1_vae.safetensors",
)
H3 = dict(
    unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors", clip="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    vae="minimax_h3_video_vae_fp16.safetensors", avae="minimax_h3_audio_vae_fp32.safetensors",
    lora="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
)

COMBOS = {
    "wan22_lightx2v":     dict(kind="wan", lora=True,  steps=4,  split=2,  cfg=1.0, shift=5.0, first_last=False, fps=16, frames=81),
    "wan22_full":         dict(kind="wan", lora=False, steps=20, split=10, cfg=3.5, shift=8.0, first_last=False, fps=16, frames=81),
    "wan22_lightx2v_flf": dict(kind="wan", lora=True,  steps=4,  split=2,  cfg=1.0, shift=5.0, first_last=True,  fps=16, frames=81),
    "anisora":            dict(kind="wan", lora=False, steps=20, split=10, cfg=3.5, shift=8.0, first_last=False, fps=16, frames=81, unet="AniSora-V3.2-I2V-A14B-{}Noise-Q8_0.gguf"),
    "anisora_lightx2v":   dict(kind="wan", lora=True,  steps=4,  split=2,  cfg=1.0, shift=5.0, first_last=False, fps=16, frames=81, unet="AniSora-V3.2-I2V-A14B-{}Noise-Q8_0.gguf"),
    "h3_turbo":           dict(kind="h3", lora=True,  steps=8,  first_last=True, fps=24, frames=124),
    "h3_full":            dict(kind="h3", lora=False, steps=20, first_last=True, fps=24, frames=124),
    "h3_turbo_i2v":       dict(kind="h3", lora=True,  steps=8,  first_last=False, fps=24, frames=124),   # 只給首幀
}


# ---------- ComfyUI API ----------
def api(path, data=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


class WS(threading.Thread):
    """極簡 websocket 客戶端（只收文字訊框），記錄各節點開始執行的時間。"""
    def __init__(self, cid):
        super().__init__(daemon=True)
        self.cid, self.events, self.stop = cid, [], False
        self.sock = socket.create_connection(("127.0.0.1", 8188))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /ws?clientId={cid} HTTP/1.1\r\nHost: 127.0.0.1:8188\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(1)
        self.sock.settimeout(1.0)

    def _read(self, n):
        b = b""
        while len(b) < n:
            try:
                c = self.sock.recv(n - len(b))
            except socket.timeout:
                if self.stop:
                    raise EOFError
                continue
            if not c:
                raise EOFError
            b += c
        return b

    def run(self):
        try:
            while not self.stop:
                h = self._read(2)
                op, ln = h[0] & 15, h[1] & 127
                if ln == 126: ln = struct.unpack(">H", self._read(2))[0]
                elif ln == 127: ln = struct.unpack(">Q", self._read(8))[0]
                payload = self._read(ln)
                if op == 1:
                    m = json.loads(payload)
                    if m["type"] in ("execution_start", "executing", "execution_success", "execution_error", "execution_cached"):
                        self.events.append((time.time(), m))
        except Exception:
            pass


class Sampler(threading.Thread):
    """每秒取樣 ComfyUI 進程樹 RSS、系統 used、swap used、nvidia-smi VRAM used。"""
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.peak = dict(comfy_rss_gb=0, sys_used_gb=0, swap_used_gb=0, vram_gb=0)
        pid = subprocess.run(["pgrep", "-f", "main.py --listen 127.0.0.1 --port 8188"], capture_output=True, text=True).stdout.split()
        self.proc = psutil.Process(int(pid[0]))

    def run(self):
        G = 1024 ** 3
        while not self.stop:
            try:
                rss = sum(p.memory_info().rss for p in [self.proc] + self.proc.children(recursive=True)) / G
                vm, sw = psutil.virtual_memory(), psutil.swap_memory()
                v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip()
                for k, val in (("comfy_rss_gb", rss), ("sys_used_gb", vm.used / G), ("swap_used_gb", sw.used / G), ("vram_gb", float(v) / 1024)):
                    self.peak[k] = max(self.peak[k], round(val, 2))
            except Exception:
                pass
            time.sleep(1)


# ---------- 輸入圖 ----------
def prep_image(src, w, h, name):
    """縮放到涵蓋 w×h，再中心裁切。"""
    im = Image.open(src).convert("RGB")
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - w) // 2, (im.height - h) // 2
    im = im.crop((l, t, l + w, t + h))
    (COMFY / "input").mkdir(exist_ok=True)
    im.save(COMFY / "input" / name)
    return name


# ---------- workflow ----------
def wf_wan(c, prompt, negative, seed, w, h, first, last, prefix):
    g = {}
    g["clip"] = dict(class_type="CLIPLoader", inputs=dict(clip_name=WAN["clip"], type="wan", device="default"))
    g["vae"] = dict(class_type="VAELoader", inputs=dict(vae_name=WAN["vae"]))
    g["pos"] = dict(class_type="CLIPTextEncode", inputs=dict(text=prompt, clip=["clip", 0]))
    g["neg"] = dict(class_type="CLIPTextEncode", inputs=dict(text=negative, clip=["clip", 0]))
    g["img0"] = dict(class_type="LoadImage", inputs=dict(image=first))
    if c["first_last"]:
        g["img1"] = dict(class_type="LoadImage", inputs=dict(image=last))
        g["i2v"] = dict(class_type="WanFirstLastFrameToVideo", inputs=dict(
            positive=["pos", 0], negative=["neg", 0], vae=["vae", 0], width=w, height=h, length=c["frames"], batch_size=1,
            start_image=["img0", 0], end_image=["img1", 0]))
    else:
        g["i2v"] = dict(class_type="WanImageToVideo", inputs=dict(
            positive=["pos", 0], negative=["neg", 0], vae=["vae", 0], width=w, height=h, length=c["frames"], batch_size=1,
            start_image=["img0", 0]))
    for tag in ("high", "low"):
        g[f"u_{tag}"] = dict(class_type="UnetLoaderGGUF", inputs=dict(unet_name=(c["unet"].format(tag.capitalize()) if "unet" in c else WAN[tag])))
        src = [f"u_{tag}", 0]
        if c["lora"]:
            g[f"l_{tag}"] = dict(class_type="LoraLoaderModelOnly", inputs=dict(model=src, lora_name=WAN[f"lora_{tag}"], strength_model=1.0))
            src = [f"l_{tag}", 0]
        g[f"ms_{tag}"] = dict(class_type="ModelSamplingSD3", inputs=dict(model=src, shift=c["shift"]))
    common = dict(cfg=c["cfg"], sampler_name="euler", scheduler="simple", steps=c["steps"])
    g["ks_high"] = dict(class_type="KSamplerAdvanced", inputs=dict(
        model=["ms_high", 0], add_noise="enable", noise_seed=seed, positive=["i2v", 0], negative=["i2v", 1], latent_image=["i2v", 2],
        start_at_step=0, end_at_step=c["split"], return_with_leftover_noise="enable", **common))
    g["ks_low"] = dict(class_type="KSamplerAdvanced", inputs=dict(
        model=["ms_low", 0], add_noise="disable", noise_seed=0, positive=["i2v", 0], negative=["i2v", 1], latent_image=["ks_high", 0],
        start_at_step=c["split"], end_at_step=10000, return_with_leftover_noise="disable", **common))
    g["dec"] = dict(class_type="VAEDecode", inputs=dict(samples=["ks_low", 0], vae=["vae", 0]))
    g["vid"] = dict(class_type="CreateVideo", inputs=dict(images=["dec", 0], fps=float(c["fps"])))
    g["save"] = dict(class_type="SaveVideo", inputs=dict(video=["vid", 0], filename_prefix=prefix, format="mp4", codec="h264"))
    return g


def wf_h3(c, prompt, seed, w, h, first, last, prefix):
    g = {}
    g["unet"] = dict(class_type="UNETLoader", inputs=dict(unet_name=H3["unet"], weight_dtype="default"))
    g["clip"] = dict(class_type="CLIPLoader", inputs=dict(clip_name=H3["clip"], type="minimax", device="default"))
    g["vae"] = dict(class_type="VAELoader", inputs=dict(vae_name=H3["vae"]))
    g["avae"] = dict(class_type="VAELoader", inputs=dict(vae_name=H3["avae"]))
    g["img0"] = dict(class_type="LoadImage", inputs=dict(image=first))
    ci = dict(clip=["clip", 0], vae=["vae", 0], prompt=prompt, width=w, height=h, length=c["frames"], first_frame=["img0", 0])
    if last:
        g["img1"] = dict(class_type="LoadImage", inputs=dict(image=last))
        ci["last_frame"] = ["img1", 0]
    g["cond"] = dict(class_type="MiniMaxH3ImageToVideo", inputs=ci)
    model = ["unet", 0]
    if c["lora"]:
        g["lora"] = dict(class_type="LoraLoaderModelOnly", inputs=dict(model=model, lora_name=H3["lora"], strength_model=1.0))
        model = ["lora", 0]
    g["sched"] = dict(class_type="BasicScheduler", inputs=dict(model=model, scheduler="simple", steps=c["steps"], denoise=1.0))
    g["samp"] = dict(class_type="KSamplerSelect", inputs=dict(sampler_name="res_multistep"))
    g["noise"] = dict(class_type="RandomNoise", inputs=dict(noise_seed=seed))
    g["guider"] = dict(class_type="BasicGuider", inputs=dict(model=model, conditioning=["cond", 0]))
    g["sca"] = dict(class_type="SamplerCustomAdvanced", inputs=dict(
        noise=["noise", 0], guider=["guider", 0], sampler=["samp", 0], sigmas=["sched", 0], latent_image=["cond", 1]))
    g["dec"] = dict(class_type="VAEDecode", inputs=dict(samples=["sca", 0], vae=["vae", 0]))
    g["adec"] = dict(class_type="VAEDecodeAudio", inputs=dict(samples=["sca", 0], vae=["avae", 0]))
    g["vid"] = dict(class_type="CreateVideo", inputs=dict(images=["dec", 0], audio=["adec", 0], fps=float(c["fps"])))
    g["save"] = dict(class_type="SaveVideo", inputs=dict(video=["vid", 0], filename_prefix=prefix, format="mp4", codec="h264"))
    return g


# ---------- 後處理 ----------
def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout


def make_sheet(mp4, png, n=8):
    nb = int(sh("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", str(mp4)).strip())
    idx = sorted({round(i * (nb - 1) / (n - 1)) for i in range(n)})
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vf", f"select='{expr}',scale=640:-1,tile=4x2", "-frames:v", "1", str(png)], check=True)
    return nb, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True, choices=list(COMBOS))  # 另可加 --variant nofade
    ap.add_argument("--shot", required=True)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--variant", default="", help="prompt 變體：nofade 會在 prompt 後加上眼瞼實體抬起、無溶接的強調；輸出到 <combo>_<variant>/")
    a = ap.parse_args()
    c = COMBOS[a.combo]
    P = yaml.safe_load((ROOT / "pilot/prompts.yaml").read_text())
    shot = P["shots"][a.shot]
    seed = a.seed or P["seed"]
    w, h = map(int, a.res.split("x"))
    if c["kind"] == "h3":   # H3 需 32 的倍數：高度 720 → 736，輸出時再中心裁回 720
        w, h = (w + 31) // 32 * 32, (h + 31) // 32 * 32
    first = prep_image(ROOT / shot["first"], w, h, f"p4_{a.shot}_first_{w}x{h}.png")
    last = prep_image(ROOT / shot["last"], w, h, f"p4_{a.shot}_last_{w}x{h}.png") if (c["first_last"] and "last" in shot) else None
    tag = a.combo + (f"_{a.variant}" if a.variant else "")
    prompt_txt = shot["prompt"].strip()
    if a.variant == "nofade":
        prompt_txt += (" The eyelids physically lift open from the top, the upper eyelid moving upward to reveal the eyes;"
                       " no crossfade, no dissolve, no ghosting, no double exposure between closed and open eyes.")
    prefix = f"video/p4_{tag}_{a.shot}"
    if c["kind"] == "wan":
        wf = wf_wan(c, prompt_txt, P["negative"].strip(), seed, w, h, first, last, prefix)
    else:
        wf = wf_h3(c, prompt_txt, seed, w, h, first, last, prefix)

    cid = uuid.uuid4().hex
    ws = WS(cid); ws.start()
    mon = Sampler(); mon.start()
    t0 = time.time()
    pid = api("/prompt", {"prompt": wf, "client_id": cid})["prompt_id"]
    status = "ok"
    err = None
    while True:
        hist = api("/history/" + pid)
        if pid in hist and hist[pid]["status"].get("completed") is not None and hist[pid]["status"].get("status_str") in ("success", "error"):
            if hist[pid]["status"]["status_str"] == "error":
                status, err = "error", json.dumps(hist[pid]["status"]["messages"][-1], ensure_ascii=False)[:3000]
            break
        time.sleep(2)
    t1 = time.time()
    time.sleep(1.5)
    mon.stop = True; ws.stop = True

    # 各節點耗時
    ev = [(t, m) for t, m in ws.events if m["type"] == "executing" and m["data"].get("prompt_id") == pid]
    per_node = {}
    for (ta, ma), (tb, _) in zip(ev, ev[1:] + [(t1, None)]):
        n = ma["data"].get("node")
        if n:
            k = f'{n}:{wf[n]["class_type"]}'
            per_node[k] = round(tb - ta, 1)
    rec = dict(combo=tag, prompt_variant=a.variant or None, prompt_used=prompt_txt if a.variant else None, shot=a.shot, status=status, error=err,
               total_s=round(t1 - t0, 1), per_node_s=per_node,
               peak=mon.peak,
               params=dict(resolution=f"{w}x{h}", frames=c["frames"], fps=c["fps"], steps=c["steps"], cfg=c.get("cfg", "BasicGuider(無 CFG)" if c["kind"] == "h3" else None),
                           shift=c.get("shift"), sampler="euler/simple" if c["kind"] == "wan" else "res_multistep/simple", seed=seed,
                           lora=c["lora"], first_last=bool(last), split_step=c.get("split")),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if status == "ok":
        out = hist[pid]["outputs"]["save"]["images"][0]
        src = COMFY / "output" / out.get("subfolder", "") / out["filename"]
        d = ROOT / "pilot" / tag
        d.mkdir(parents=True, exist_ok=True)
        mp4 = d / f"{a.shot}.mp4"
        vf = f"crop={OUT_W}:{OUT_H}" if h != OUT_H or w != OUT_W else "null"
        sh("ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", "-vf", vf, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(mp4))
        nb, idx = make_sheet(mp4, d / f"{a.shot}_sheet.png")
        rec.update(output=str(mp4.relative_to(ROOT)), frames_out=nb, sheet_frame_idx=idx)
    mp = ROOT / "pilot/metrics.json"
    M = json.loads(mp.read_text()) if mp.exists() else {}
    M[f"{tag}/{a.shot}"] = rec
    mp.write_text(json.dumps(M, ensure_ascii=False, indent=2))
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
