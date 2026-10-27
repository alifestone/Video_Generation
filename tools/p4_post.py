#!/usr/bin/env python3
"""P4 後製技術測試：RIFE（ComfyUI 節點）、Real-ESRGAN（獨立 spandrel，逐幀計時與 VRAM）。
~/creative/ComfyUI/comfyui/bin/python tools/p4_post.py rife <in.mp4> <out_24.mp4>
~/creative/ComfyUI/comfyui/bin/python tools/p4_post.py esrgan <in.mp4> <out_1080.mp4> [<lanczos_ref.mp4>]
"""
import json, subprocess, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent

def probe(mp4):
    o = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,r_frame_rate,nb_frames","-of","json",str(mp4)],capture_output=True,text=True).stdout
    s = json.loads(o)["streams"][0]; n,d = map(int,s["r_frame_rate"].split("/")); return s["width"],s["height"],n/d,int(s.get("nb_frames",0))

def rife(src, dst, mult=3):
    import p4_video as P
    wf = {
      "ld": dict(class_type="VHS_LoadVideoPath", inputs=dict(video=str(Path(src).resolve()), force_rate=0, custom_width=0, custom_height=0, frame_load_cap=0, skip_first_frames=0, select_every_nth=1, format="None")),
      "rife": dict(class_type="RIFE VFI", inputs=dict(ckpt_name="rife49.pth", frames=["ld",0], clear_cache_after_n_frames=10, multiplier=mult, fast_mode=True, ensemble=True, scale_factor=1.0, dtype="float32", torch_compile=False, batch_size=1)),
      "vid": dict(class_type="CreateVideo", inputs=dict(images=["rife",0], fps=16.0*mult)),
      "save": dict(class_type="SaveVideo", inputs=dict(video=["vid",0], filename_prefix="video/p4_rife", format="mp4", codec="h264")),
    }
    cid = uuid.uuid4().hex; mon = P.Sampler(); mon.start(); t0 = time.time()
    pid = P.api("/prompt", {"prompt": wf, "client_id": cid})["prompt_id"]
    while True:
        h = P.api("/history/"+pid)
        if pid in h and h[pid]["status"].get("status_str") in ("success","error"): break
        time.sleep(2)
    t1 = time.time(); mon.stop = True
    if h[pid]["status"]["status_str"] == "error": print(json.dumps(h[pid]["status"]["messages"][-1])[:3000]); sys.exit(1)
    o = h[pid]["outputs"]["save"]["images"][0]; mid = P.COMFY/"output"/o.get("subfolder","")/o["filename"]
    full48 = Path(dst).with_name(Path(dst).stem.replace("_24","")+"_48.mp4")
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(mid),"-c:v","libx264","-crf","14","-pix_fmt","yuv420p",str(full48)],check=True)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(mid),"-vf","select='not(mod(n\\,2))',setpts=N/(24*TB)","-r","24","-c:v","libx264","-crf","14","-pix_fmt","yuv420p",str(dst)],check=True)
    print(json.dumps(dict(total_s=round(t1-t0,1), peak=mon.peak, out48=str(full48), out24=str(dst), in_info=probe(src), out48_info=probe(full48), out24_info=probe(dst))))

def esrgan(src, dst, ref):
    import numpy as np, torch, cv2
    from spandrel import ModelLoader
    W,H,fps,_ = probe(src)
    m = ModelLoader().load_from_file(str(Path.home()/"creative/ComfyUI/models/upscale_models/RealESRGAN_x4plus_anime_6B.pth")).model.cuda().half().eval()
    OW,OH = 1920,1080
    rd = subprocess.Popen(["ffmpeg","-v","error","-i",str(src),"-f","rawvideo","-pix_fmt","rgb24","-"],stdout=subprocess.PIPE)
    wr = subprocess.Popen(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{OW}x{OH}","-r",str(fps),"-i","-","-c:v","libx264","-crf","14","-pix_fmt","yuv420p",str(dst)],stdin=subprocess.PIPE)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(src),"-vf",f"scale={OW}:{OH}:flags=lanczos","-c:v","libx264","-crf","14","-pix_fmt","yuv420p",str(ref)],check=True)
    torch.cuda.reset_peak_memory_stats(); times=[]; tile=0
    def up(x):
        nonlocal tile
        if tile==0:
            try: return m(x)
            except torch.OutOfMemoryError: torch.cuda.empty_cache(); tile=512
        # 分塊（重疊 32 px）
        _,_,h,w = x.shape; out = torch.zeros(1,3,h*4,w*4,device=x.device,dtype=x.dtype); pad=32; t=tile
        for y in range(0,h,t):
            for xx in range(0,w,t):
                y0,x0=max(0,y-pad),max(0,xx-pad); y1,x1=min(h,y+t+pad),min(w,xx+t+pad)
                o=m(x[:,:,y0:y1,x0:x1]); cy,cx=(y-y0)*4,(xx-x0)*4; hh,ww=min(t,h-y)*4,min(t,w-xx)*4
                out[:,:,y*4:y*4+hh,xx*4:xx*4+ww]=o[:,:,cy:cy+hh,cx:cx+ww]
        return out
    n=0
    while True:
        b = rd.stdout.read(W*H*3)
        if len(b)<W*H*3: break
        x = torch.from_numpy(np.frombuffer(b,np.uint8).reshape(H,W,3).copy()).cuda().permute(2,0,1)[None].half()/255
        torch.cuda.synchronize(); t0=time.time()
        with torch.no_grad(): y = up(x)
        torch.cuda.synchronize(); times.append(time.time()-t0)
        y = (y.clamp(0,1)[0].permute(1,2,0).float().cpu().numpy()*255+.5).astype(np.uint8)
        y = cv2.resize(y,(OW,OH),interpolation=cv2.INTER_AREA)
        wr.stdin.write(y.tobytes()); n+=1
    wr.stdin.close(); wr.wait(); rd.wait()
    print(json.dumps(dict(frames=n, per_frame_s_mean=round(float(np.mean(times)),3), per_frame_s_first=round(times[0],3), per_frame_s_median=round(float(np.median(times)),3),
                          total_s=round(float(sum(times)),1), peak_vram_torch_gb=round(torch.cuda.max_memory_allocated()/2**30,2),
                          peak_vram_reserved_gb=round(torch.cuda.max_memory_reserved()/2**30,2), tiled=tile)))

if __name__=="__main__":
    c=sys.argv[1]
    if c=="rife": rife(sys.argv[2],sys.argv[3])
    elif c=="esrgan": esrgan(sys.argv[2],sys.argv[3],sys.argv[4])
