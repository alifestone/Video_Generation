#!/usr/bin/env python3
"""s002 路線 C：深度圖＋自寫視差推近（反向取樣 grid_sample）。
~/creative/ComfyUI/comfyui/bin/python tools/p4_parallax.py <keyframe.png> <depth.png> <out.mp4> [--frames 54 --fps 24 --zoom 0.05]
每像素放大率 s = 1 + e(t) * (z_far + (z_near-z_far)*d)，d 為正規化深度（近=1）。平均推近約 zoom（以畫面平均 s 計）。
"""
import argparse, subprocess, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("kf"); ap.add_argument("depth"); ap.add_argument("out")
ap.add_argument("--frames", type=int, default=54); ap.add_argument("--fps", type=int, default=24); ap.add_argument("--zoom", type=float, default=0.05)
ap.add_argument("--spread", type=float, default=0.6, help="近遠放大率差（相對 zoom）"); ap.add_argument("--blur", type=int, default=9)
A = ap.parse_args()
OW, OH = 1920, 1080
def cover(im, resample):
    s = max(OW / im.width, OH / im.height); im = im.resize((round(im.width*s), round(im.height*s)), resample)
    l, t = (im.width-OW)//2, (im.height-OH)//2; return im.crop((l, t, l+OW, t+OH))
img = torch.from_numpy(np.asarray(cover(Image.open(A.kf).convert("RGB"), Image.LANCZOS)).copy()).permute(2,0,1)[None].float().cuda()/255
d = torch.from_numpy(np.asarray(cover(Image.open(A.depth).convert("L"), Image.BICUBIC)).copy()).float().cuda()[None,None]/255
lo, hi = torch.quantile(d, 0.01), torch.quantile(d, 0.99); d = ((d-lo)/(hi-lo)).clamp(0,1)
k = A.blur*2+1; d = F.avg_pool2d(F.pad(d,(k//2,)*4,mode="replicate"), k, 1)   # 平滑深度避免邊緣撕裂
mean_d = float(d.mean()); z_far = A.zoom*(1-A.spread*mean_d/ max(1e-6,1)) if False else None
# 讓平均放大率 = zoom：z_far + (z_near-z_far)*mean_d = zoom，z_near - z_far = spread*zoom
dz = A.spread*A.zoom; z_far = A.zoom - dz*mean_d; z_near = z_far + dz
yy, xx = torch.meshgrid(torch.linspace(-1,1,OH,device="cuda"), torch.linspace(-1,1,OW,device="cuda"), indexing="ij")
base = torch.stack([xx,yy],-1)[None]
enc = subprocess.Popen(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{OW}x{OH}","-r",str(A.fps),"-i","-","-c:v","libx264","-crf","12","-pix_fmt","yuv420p",A.out],stdin=subprocess.PIPE)
N = A.frames
for i in range(N):
    t = i/(N-1); e = t*t*(3-2*t)    # smoothstep 緩入緩出
    s = 1 + e*(z_far + (z_near-z_far)*d[0,0])
    grid = base / s[None,:,:,None]
    out = F.grid_sample(img, grid, mode="bicubic", padding_mode="border", align_corners=True).clamp(0,1)
    enc.stdin.write((out[0].permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait()
print(json.dumps(dict(frames=N, fps=A.fps, dur_s=round(N/A.fps,3), zoom_mean_end=A.zoom, z_far=round(z_far,4), z_near=round(z_near,4), mean_depth=round(mean_d,3))))
