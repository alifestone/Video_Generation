#!/usr/bin/env python3
"""聽感確認用音檔：原曲疊加節拍 click（首拍高而重、其餘拍低而輕），段落邊界加一聲長提示音。

uv run --no-project --python 3.12 --with numpy --with soundfile python tools/make_click_check.py \
    --audio audio/song.wav --timeline analysis/timeline.json -o analysis/check/song_clicks.wav

聽什麼：click 是否貼著鼓點、首拍是否在小節頭；每個低沉長音（330Hz）處是否為段落轉換；若有 alt_bar 則以 pip-pip 雙音標示備選邊界；hits（如 drop）在小節首拍加一個上升三連音 blip（600→900→1400Hz），聽它是否貼著重音。
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf


def burst(sr, freq, dur, gain, decay=40.0):
    t = np.arange(int(sr * dur)) / sr
    return (gain * np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)).astype(np.float32)


def add(buf, t, sig, sr):
    i = int(round(t * sr))
    if 0 <= i < len(buf):
        j = min(len(buf), i + len(sig))
        buf[i:j] += sig[: j - i, None] if buf.ndim == 2 else sig[: j - i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("-o", "--output", default="analysis/check/song_clicks.wav")
    ap.add_argument("--song-gain", type=float, default=0.8)
    ap.add_argument("--click-gain", type=float, default=0.5)
    args = ap.parse_args()

    y, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    y = y * args.song_gain
    tl = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    g = args.click_gain
    down, other = burst(sr, 1800, 0.06, g), burst(sr, 900, 0.04, g * 0.45)
    for b in tl["beats"]:
        add(y, b["t"], down if b["is_downbeat"] else other, sr)
    # 段落邊界（timeline.sections，即手動覆寫後的段落）：低音 330Hz 長音 0.35s
    mark = burst(sr, 330, 0.35, g * 1.2, decay=6.0)
    for s in tl["sections"][1:]:
        add(y, s["t_start"], mark, sr)
    # 備選邊界（alt_bar）：兩聲短促高音 pip-pip（1200Hz），與主邊界的低沉長音區分
    pip = burst(sr, 1200, 0.07, g, decay=25.0)
    for s in tl["sections"]:
        if "alt_t" in s:
            add(y, s["alt_t"], pip, sr)
            add(y, s["alt_t"] + 0.12, pip, sr)
    # hits：上升三連音 blip（600/900/1400Hz，每音 0.06s），對齊該小節首拍
    blip = np.concatenate([burst(sr, f, 0.06, g * 1.1, decay=15.0) for f in (600, 900, 1400)])
    for h in tl.get("hits", []):
        add(y, h["t"], blip, sr)
    peak = float(np.abs(y).max())
    if peak > 0.99:
        y *= 0.99 / peak
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, sr, subtype="PCM_16")
    print(f"wrote {out}  ({len(tl['beats'])} clicks, {len(tl['sections']) - 1} section marks, {len(tl.get('hits', []))} hits, bpm {tl['meta']['bpm']:g})")
    print("cue list (time  label  bar  [snap / alt])  -- tone = low long tone, alt = pip-pip:")
    cues = []
    for s in tl["sections"]:
        cues.append((s["t_start"], f"{s['label']:<14} bar {s['bar_start']:<3} tone  snap {s['snap_beats']:+.2f} beats"))
        if "alt_t" in s:
            cues.append((s["alt_t"], f"{s['label']:<14} bar {s['alt_bar']:<3} pip-pip (alt)"))
    for h in tl.get("hits", []):
        cues.append((h["t"], f"{h['type']:<14} bar {h['bar']:<3} rising blip (hit)"))
    for t, txt in sorted(cues):
        print(f"  {int(t // 60)}:{t % 60:05.2f}  {txt}")

if __name__ == "__main__":
    sys.exit(main())
