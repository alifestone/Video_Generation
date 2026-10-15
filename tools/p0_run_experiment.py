#!/usr/bin/env python3
"""P0 實驗：以 ComfyUI HTTP API 依序跑 6 目標 x 2 輸入模式。
讀 workflows/p0_ref_gen.json（API 格式，含佔位 id），逐一填入 prompt/seed/圖片後送出。"""
import json, time, subprocess, threading, urllib.request, urllib.parse, uuid, sys, re
from pathlib import Path

HOST = "http://127.0.0.1:8188"
root = Path(__file__).resolve().parent.parent
wf0 = json.loads((root / "workflows/p0_ref_gen.json").read_text())
STEPS, CFG, SAMPLER, SCHED = 35, 4.0, "euler", "simple"

COMMON = ("Single character only, no text, no letters, no captions, no symbols, no stars, no manga effect marks, "
          "no panel frames, plain pure white background, clean anime character-sheet illustration style identical to the reference. "
          "Keep the exact same character and outfit as the reference: same face, long purple braided high ponytail with a silver chain, "
          "white sleeveless high-collar zip-up crop top, white cargo mini skirt with black belt, black thigh strap on the same thigh as in the reference "
          "(the thigh on the left side of the image when facing the viewer), silver bracelets, white chunky sneakers; correct hands with exactly five fingers each. ")
TARGETS = [
 ("full_front", "full_front", 704, 1024, "Redraw the character as a full-body standing pose seen from the front, whole body from head to shoes visible, neutral relaxed stance, arms at sides. "),
 ("side", "side", 704, 1024, "Redraw the character as a full-body standing pose seen from the side (profile view, facing left), whole body from head to shoes visible, arms relaxed. "),
 ("back", "side", 704, 1024, "Redraw the character as a full-body standing pose seen from directly behind (back view), whole body from head to shoes visible, the long braided ponytail with silver chain visible down her back. "),
 ("expr_happy", "expr_happy", 832, 1024, "Redraw the character as a head-and-shoulders portrait with a happy open-mouth smile, bright cheerful expression, facing the viewer. "),
 ("expr_angry", "expr_angry", 832, 1024, "Redraw the character as a head-and-shoulders portrait with an angry expression, furrowed eyebrows, arms crossed, facing the viewer. "),
 ("action_point", "action_point", 832, 1024, "Redraw the character as an upper-body action pose, confidently pointing one index finger toward the viewer with a cheerful expression. "),
]
SEED0 = 20251101

def api(path, data=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def upload(p):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{p.name}"\r\nContent-Type: image/png\r\n\r\n').encode() \
           + p.read_bytes() + (f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{b}--\r\n').encode()
    r = urllib.request.Request(HOST + "/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(r))["name"]

peak = {"ram_used_mb": 0, "swap_used_mb": 0, "vram_mb": 0}
stop = False
def monitor():
    while not stop:
        try:
            m = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.split("\n")
            ram = int(m[1].split()[2]); sw = int(m[2].split()[2])
            v = max(int(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True).stdout.split())
            peak["ram_used_mb"] = max(peak["ram_used_mb"], ram); peak["swap_used_mb"] = max(peak["swap_used_mb"], sw); peak["vram_mb"] = max(peak["vram_mb"], v)
        except Exception: pass
        time.sleep(2)

def run(mode, idx, tgt):
    name, crop, w, h, ptxt = tgt
    wf = json.loads(json.dumps(wf0))
    sheet = upload(root / "refs/動畫_女生_LUNA.jpg")
    wf["10"]["inputs"]["image"] = sheet
    prompt = "Image 1 is a character reference sheet of the character. " + (
        "Image 2 is a cropped reference cell of the same character showing the target view/expression. " if mode == "B" else "") + ptxt + COMMON
    wf["20"]["inputs"]["prompt"] = prompt
    if mode == "B":
        wf["11"]["inputs"]["image"] = upload(root / f"refs/crops/{crop}.png")
    else:
        del wf["11"]
        del wf["20"]["inputs"]["image2"]; del wf["21"]["inputs"]["image2"]
    wf["30"]["inputs"].update(width=w, height=h)
    seed = SEED0 + [t[0] for t in TARGETS].index(name)
    wf["40"]["inputs"].update(seed=seed, steps=STEPS, cfg=CFG, sampler_name=SAMPLER, scheduler=SCHED)
    wf["60"]["inputs"]["filename_prefix"] = f"p0exp/{mode}_{idx:02d}_{name}"
    t0 = time.time()
    pid = api("/prompt", {"prompt": wf, "client_id": "p0"})["prompt_id"]
    while True:
        h_ = api("/history/" + pid)
        if pid in h_ and h_[pid].get("outputs"): break
        if pid in h_ and h_[pid]["status"].get("status_str") == "error": raise RuntimeError(json.dumps(h_[pid]["status"])[:2000])
        time.sleep(3)
    dt = time.time() - t0
    im = h_[pid]["outputs"]["60"]["images"][0]
    data = urllib.request.urlopen(f"{HOST}/view?" + urllib.parse.urlencode(im)).read()
    d = root / "refs/experiment" / ("A_sheet_only" if mode == "A" else "B_sheet_plus_crop")
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{idx:02d}_{name}.png"; out.write_bytes(data)
    rec = dict(mode=mode, idx=idx, target=name, file=str(out.relative_to(root)), seed=seed, steps=STEPS, cfg=CFG,
               sampler=SAMPLER, scheduler=SCHED, size=f"{w}x{h}", seconds=round(dt, 1), prompt=prompt)
    print(rec, flush=True)
    return rec

if __name__ == "__main__":
    only = sys.argv[1:]  # 例如 A:1 限制只跑某些
    threading.Thread(target=monitor, daemon=True).start()
    recs = []
    for i, t in enumerate(TARGETS, 1):
        for mode in "AB":
            if only and f"{mode}:{i}" not in only: continue
            recs.append(run(mode, i, t))
            (root / "refs/experiment/results.json").write_text(json.dumps(dict(records=recs, peak=peak), ensure_ascii=False, indent=1))
    stop = True
    print("PEAK", peak)
