#!/usr/bin/env python3
"""驗證 pilot/intro_pilot.mp4：每個拍點事件截圖＋幀差／區域指標量測拍點誤差 → pilot/post_checks/beat_errors.json
~/creative/ComfyUI/comfyui/bin/python tools/p4_intro_checks.py"""
import json, subprocess, numpy as np, cv2
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
CHK = ROOT / "pilot/post_checks"; P = json.load(open(CHK / "plan.json")); F0 = P["frames"][0]
b = subprocess.run(["ffmpeg","-v","error","-i",str(ROOT/"pilot/intro_pilot.mp4"),"-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
V = np.frombuffer(b, np.uint8).reshape(-1, 1080, 1920, 3); print(V.shape)
sc = json.load(open(ROOT/"shots/s002/screens.json"))["screens"]
def abs_(i): return F0 + i
res = {}
# s002：每組螢幕（鏡頭有視差，bbox 以 1344→1920 縮放近似＋容忍 5%）亮起幀＝區域平均亮度首次 > 基準 3 倍且 > 20
def reg(r, c):
    x0,y0,x1,y1 = sc[f"{r},{c}"]; s=1920/1344
    return int((x0+8)*s), int((y0+8-6)*s), int((x1-8)*s), int((y1-8-6)*s)
cols_groups = {0:[(0,3),(2,3)], 1:[(r,c) for r in range(3) for c in (2,4)], 2:[(r,c) for r in range(3) for c in (1,5)], 3:[(r,c) for r in range(3) for c in (0,6)]}
for g, cells in cols_groups.items():
    ev = P["screen_beats"][g]; lum=[]; sat=[]
    for i in range(54):
        L=[];S=[]
        for (r,c) in cells:
            x0,y0,x1,y1 = reg(r,c); R=V[i,y0:y1,x0:x1].astype(np.float32)
            L.append(R.mean()); hsv=cv2.cvtColor(V[i,y0:y1,x0:x1],cv2.COLOR_RGB2HSV); S.append(hsv[...,1].mean())
        lum.append(np.mean(L)); sat.append(np.mean(S))
    lum=np.array(lum); sat=np.array(sat)
    first=next(i for i in range(54) if lum[i]>25)           # 亮起
    # 上一幀必須是暗的（g=0 時第一幀 i=0 無前幀，另以 keyframe 暗態比對）
    gray_done = next((i for i in range(first,54) if sat[i] < 0.25*sat[first:first+3].max()), None)
    res[f"s002_beat{g+1}"] = dict(planned=ev, measured=abs_(first), err=abs_(first)-ev, lum_prev=float(lum[first-1]) if first else None, lum_first=float(lum[first]),
                                  gray_done=abs_(gray_done) if gray_done is not None else None, next_beat=(P["screen_beats"][g+1] if g<3 else P["s003"][0]))
# s003 b2：右緣條帶（x>1850）相對 s003 首幀變化
s3=P["s003"][0]-F0; ref=V[s3,:, 1850:].astype(np.int16)
d=[np.abs(V[i,:,1850:].astype(np.int16)-ref).mean() for i in range(s3,s3+54)]
first=next(i for i,x in enumerate(d) if x>1.0); res["s003_echo_in"]=dict(planned=P["echo_in"], measured=P["s003"][0]+first, err=P["s003"][0]+first-P["echo_in"], series=[round(float(x),2) for x in d[10:22]])
# s003 b3：ECHO 停穩（光帶中心 x 逐幀位移 < 2px）、ring 青色像素首現、glow 峰值
def ring_px(i):
    im=V[i].astype(np.int16); m=(im[...,0]<120)&(im[...,1]>170)&(im[...,2]>210); m[:, 1300:]=False; return int(m.sum())
rp=[ring_px(s3+j) for j in range(54)]; base=rp[10]
ring_first=next(j for j in range(54) if rp[j]>base+3000)
def band_x(i):
    im=V[i].astype(np.int16); m=(im[...,0]<120)&(im[...,1]>180)&(im[...,2]>215); m[:, :1300]=False; m[:60]=False; m[520:]=False
    ys,xs=np.nonzero(m); return (xs.mean() if len(xs)>80 else None)
bx=[band_x(s3+j) for j in range(54)]
def white_right(i):
    im=V[i].astype(np.int16); w=(im.min(-1)>215); w[:, :1330]=False; w[:60]=False; w[520:]=False
    ys,xs=np.nonzero(w); return (xs.max() if len(xs)>2000 else None)
wr=[white_right(s3+j) for j in range(54)]
pos_stop=next(j for j in range(14,54) if wr[j] and wr[j-1] and wr[j]<1900 and all(wr[q] and abs(wr[q]-wr[q-1])<=2 for q in range(j,min(j+4,54))))
bxf=bx[-1]
face=next(j for j in range(1,54) if bx[j] is not None and all(bx[q] is not None and abs(bx[q]-bxf)<=4 for q in range(j,54)))
bright=[]
for j in range(54):
    im=V[s3+j,60:520,1300:].astype(np.int16); bright.append(float(((im[...,0]<160)&(im[...,1]>200)&(im[...,2]>230)).sum()))
glow_pk=int(np.argmax(bright))
res["s003_ring_in"]=dict(planned=P["echo_stop_ring"], measured=P["s003"][0]+ring_first, err=P["s003"][0]+ring_first-P["echo_stop_ring"], ring_px_series=rp[26:34])
res["s003_ring_full"]=dict(planned=P["echo_stop_ring"]+3, measured=P["s003"][0]+next(j for j in range(ring_first,54) if rp[j]>=0.97*max(rp)), )
res["s003_echo_stop_pos"]=dict(planned=P["echo_stop_ring"], measured=P["s003"][0]+pos_stop, err=P["s003"][0]+pos_stop-P["echo_stop_ring"], right_edge_x=wr[10:32])
res["s003_echo_face_front"]=dict(planned=P["echo_stop_ring"], measured=P["s003"][0]+face, err=P["s003"][0]+face-P["echo_stop_ring"], band_x=[None if x is None else round(float(x),1) for x in bx[20:32]])
res["s003_glow_peak"]=dict(planned=P["echo_stop_ring"]+1, measured=P["s003"][0]+glow_pk)
# s004：睜眼起點：眼區相對前一幀差 > 4（排除前段的緩慢頭部動作）
s4=P["s004"][0]-F0; win=V[s4:s4+55,450:720,150:1770].astype(np.int16)   # 1080p 眼區
dd=[0]+[float(np.abs(win[i]-win[i-1]).mean()) for i in range(1,55)]
first=next(i for i in range(1,55) if dd[i]>9)
res["s004_eye_open"]=dict(planned=P["eye_open"], measured=P["s004"][0]+first, err=P["s004"][0]+first-P["eye_open"], diff_series=[round(x,2) for x in dd[20:32]], pre_roll_max_diff=round(max(dd[1:first]),2))
json.dump(res, open(CHK/"beat_errors.json","w"), indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o,"item") else str(o))
for k,v in res.items(): print(k, {a:b for a,b in v.items() if not a.endswith("series")})
# 驗證圖
def panel(frames, name, labels=None, crop=None):
    ims=[]
    for f in frames:
        im=V[f-F0].copy()
        if crop: x0,y0,x1,y1=crop; im=im[y0:y1,x0:x1]
        im=cv2.resize(im,(640,int(640*im.shape[0]/im.shape[1])))
        cv2.putText(im,f"frame {f} (+{f-F0})",(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2); ims.append(im)
    cv2.imwrite(str(CHK/name), cv2.cvtColor(np.hstack(ims),cv2.COLOR_RGB2BGR))
for g in range(4):
    e=P["screen_beats"][g]; panel([max(F0,e-1),e,e+2,e+5,e+9] if g>0 else [e,e+2,e+3,e+6,e+9], f"s002_beat{g+1}_f{e}.png")
e=P["echo_in"]; panel([e-1,e,e+2,e+5,e+9],f"s003_echo_in_f{e}.png")
e=P["echo_stop_ring"]; panel([e-1,e,e+1,e+3,e+6],f"s003_ring_glow_f{e}.png")
e=P["eye_open"]; panel([e-1,e,e+2,e+5,e+9,e+14],f"s004_eye_open_f{e}.png")
panel([P["eye_open"]+16, P["eye_open"]+30],"s004_pupil_ring.png")
panel([P["eye_open"]+16],"s004_pupil_ring_crop.png",crop=(150,300,1130,560))
