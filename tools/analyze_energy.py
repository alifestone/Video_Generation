#!/usr/bin/env python3
"""P1-B：以 librosa 分析能量 (RMS)、onset 強度、速度候選與拍點，輸出逐影片幀 (24fps) 的 analysis/librosa.json。

執行（拋棄式 uv 環境）：
  uv run --no-project --python 3.12 --with librosa==0.11.0 --with soundfile \
      python tools/analyze_energy.py audio/song.wav -o analysis/librosa.json
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

root = Path(__file__).resolve().parent.parent
HOP = 512
FPS = 24


def sig(x, n=6):
    return [float(f"{v:.{n}g}") for v in np.asarray(x, dtype=float)]


def to_video_frames(values, times, n_frames):
    """每個影片幀 k 取 [k/FPS,(k+1)/FPS) 內 hop 中心時間的平均；空窗則內插。"""
    idx = np.floor(times * FPS).astype(int)
    out = np.full(n_frames, np.nan)
    ok = (idx >= 0) & (idx < n_frames)
    sums = np.bincount(idx[ok], weights=values[ok], minlength=n_frames)[:n_frames]
    cnts = np.bincount(idx[ok], minlength=n_frames)[:n_frames]
    has = cnts > 0
    out[has] = sums[has] / cnts[has]
    if not has.all():
        k = np.arange(n_frames)
        out[~has] = np.interp(k[~has], k[has], out[has])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=root / "analysis" / "librosa.json")
    a = ap.parse_args()

    y, sr = sf.read(a.audio, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    duration = len(y) / sr
    n_frames = round(duration * FPS)

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    t_rms = librosa.times_like(rms, sr=sr, hop_length=HOP)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    t_on = librosa.times_like(onset, sr=sr, hop_length=HOP)
    rms_f = to_video_frames(rms, t_rms, n_frames)
    on_f = to_video_frames(onset, t_on, n_frames)

    # 速度分布：tempogram 時間平均，取前 5 個峰值
    tg = librosa.feature.tempogram(onset_envelope=onset, sr=sr, hop_length=HOP)
    dist = tg.mean(axis=1)
    bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=HOP)
    pk = [i for i in range(1, len(dist) - 1)
          if dist[i] > dist[i - 1] and dist[i] >= dist[i + 1] and 40 <= bpms[i] <= 300]
    pk = sorted(pk, key=lambda i: -dist[i])[:5]
    cands = [{"bpm": float(f"{bpms[i]:.4g}"), "score": float(f"{dist[i] / dist[pk[0]]:.4g}")} for i in pk]

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=HOP)
    beat_t = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)

    doc = {
        "source": str(a.audio),
        "source_sha256": hashlib.sha256(a.audio.read_bytes()).hexdigest(),
        "librosa_version": librosa.__version__,
        "sr": int(sr),
        "hop_length": HOP,
        "duration_sec": float(f"{duration:.9g}"),
        "fps": FPS,
        "n_frames": int(n_frames),
        "rms": sig(rms_f),
        "onset_strength": sig(on_f),
        "rms_norm": sig(rms_f / rms_f.max()),
        "onset_norm": sig(on_f / on_f.max()),
        "tempo_candidates": cands,
        "librosa_beats_sec": sig(beat_t, 6),
        "librosa_bpm": float(f"{float(np.atleast_1d(tempo)[0]):.6g}"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(",\n".join(f'"{k}": {json.dumps(v, separators=(",", ":"))}' for k, v in doc.items()))
        f.write("\n}\n")
    print("wrote", a.out, f"{a.out.stat().st_size/1024:.0f} KB", "frames", n_frames, "bpm", doc["librosa_bpm"])


if __name__ == "__main__":
    sys.exit(main())
