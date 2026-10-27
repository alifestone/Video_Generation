#!/usr/bin/env python3
"""P4 分析工具：相鄰幀差異時間序列、尖峰偵測、密集取樣 sheet。
python3 tools/p4_analyze.py series <mp4> [--roi x0,y0,x1,y1(比例0-1)]
python3 tools/p4_analyze.py open <mp4> <out.png> [--center N] [--step 2] [--sec 1.0] [--cols 5]
python3 tools/p4_analyze.py frames <mp4> <out.png> <start> <end> [--step 1] [--cols 6] [--w 320]
"""
import argparse, json, subprocess, sys
import numpy as np

def probe(mp4):
    o = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,r_frame_rate","-of","json",mp4],capture_output=True,text=True).stdout
    s = json.loads(o)["streams"][0]; n,d = map(int,s["r_frame_rate"].split("/"))
    return s["width"], s["height"], n/d

def load_gray(mp4, w=480):
    W,H,fps = probe(mp4); h = round(H*w/W)//2*2
    raw = subprocess.run(["ffmpeg","-v","error","-i",mp4,"-vf",f"scale={w}:{h}:flags=area","-f","rawvideo","-pix_fmt","gray","-"],capture_output=True).stdout
    a = np.frombuffer(raw,np.uint8).reshape(-1,h,w).astype(np.float32)
    return a, fps

def series(mp4, roi=None):
    a,fps = load_gray(mp4)
    if roi:
        x0,y0,x1,y1 = roi; H,W = a.shape[1:]
        a = a[:, int(y0*H):int(y1*H), int(x0*W):int(x1*W)]
    d = np.abs(np.diff(a,axis=0)).mean((1,2))
    med = np.median(d); mad = np.median(np.abs(d-med))+1e-6
    z = (d-med)/(1.4826*mad)
    spikes = [(int(i+1), round(float(d[i]),3), round(float(z[i]),1)) for i in np.where((z>6)&(d>2*med+0.05))[0]]
    return dict(frames=len(a), fps=fps, diff_mean=round(float(d.mean()),4), diff_median=round(float(med),4), diff_max=round(float(d.max()),4),
                diff_max_at=int(d.argmax()+1), spikes=spikes, diff=[round(float(x),3) for x in d]), a

def sheet(mp4, out, idx, cols, w):
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    rows = -(-len(idx)//cols)
    # drawtext 不一定可用，改輸出純 tile，幀號寫在 stdout
    subprocess.run(["ffmpeg","-y","-v","error","-i",mp4,"-vf",f"select='{expr}',scale={w}:-1,tile={cols}x{rows}:padding=2","-frames:v","1",out],check=True)
    print("frame idx (row-major):", idx)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("mp4"); ap.add_argument("out", nargs="?")
    ap.add_argument("rest", nargs="*"); ap.add_argument("--roi"); ap.add_argument("--center",type=int); ap.add_argument("--step",type=int,default=2)
    ap.add_argument("--sec",type=float,default=1.0); ap.add_argument("--cols",type=int,default=5); ap.add_argument("--w",type=int,default=384)
    A = ap.parse_args()
    if A.cmd == "series":
        r,_ = series(A.mp4, tuple(map(float,A.roi.split(","))) if A.roi else None); r.pop("diff") if False else None
        print(json.dumps({k:v for k,v in r.items() if k!="diff"},ensure_ascii=False)); print("diff:",r["diff"])
    elif A.cmd == "open":
        r,_ = series(A.mp4); d = np.array(r["diff"]); fps = r["fps"]
        # 睜眼中心：平滑後的運動能量重心（去掉首尾 3 幀）
        k = np.convolve(d,np.ones(5)/5,mode="same"); c = A.center if A.center is not None else int(k[3:-3].argmax()+3)
        half = round(A.sec*fps); lo = max(0,c-half); hi = min(r["frames"]-1,c+half)
        idx = list(range(lo,hi+1,A.step)); print("center",c,"window",lo,hi,"fps",fps)
        sheet(A.mp4,A.out,idx,A.cols,A.w)
    elif A.cmd == "frames":
        s,e = int(A.rest[0]),int(A.rest[1]); sheet(A.mp4,A.out,list(range(s,e+1,A.step)),A.cols,A.w)
