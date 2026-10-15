#!/usr/bin/env python3
"""P1 驗證：以 demucs 分軌（drums / bass / other / vocals）客觀檢查 allin1 拍點與小節相位，並產出聽感輔助檔。

執行（拋棄式 uv 環境；分軌不存在時會用 env-allin1 的 demucs 自動重跑，約 30 秒）：
  uv run --no-project --python 3.12 --with librosa==0.11.0 --with soundfile --with matplotlib --with scipy \
      python tools/verify_beats.py --audio audio/song.wav --timeline analysis/timeline.json \
      --out analysis/check

輸出（皆在 gitignored 的 analysis/check/ 下）：
  verify_report.json      各問題的數字結果
  beat_corrections.json   Q2 建議的修正拍點（不會改 timeline.json）
  excerpts/NN_*.wav       可疑處的 ~10s 原曲+click 片段（click 方案同 make_click_check.py）
  plots/NN_*.png          drum/kick/bass onset 包絡 + 拍線（粗線=小節首拍）
唯讀：不修改 timeline.json / sections.json / struct。
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
import librosa
import scipy.signal as ss
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_click_check import burst, add  # noqa: E402

SR, HOP = 22050, 128
DEMUCS_PY = Path.home() / "creative/env-allin1/.venv/bin/python"


# ---------------------------------------------------------------- features
def ensure_stems(audio, stem_root):
    d = stem_root / "song"
    if not (d / "drums.wav").exists():
        print("stems missing -> running demucs htdemucs (cpu) ...")
        subprocess.run([str(DEMUCS_PY), "-m", "demucs", "-n", "htdemucs", "-d", "cpu", "-o", str(stem_root / "_tmp"), str(audio)], check=True)
        d.mkdir(parents=True, exist_ok=True)
        for n in ("drums", "bass", "other", "vocals"):
            (stem_root / "_tmp/htdemucs/song" / f"{n}.wav").replace(d / f"{n}.wav")
    return d


def env_of(x, fmax=None, n_mels=64):
    oe = librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP, n_fft=1024, n_mels=n_mels,
                                      fmax=fmax or SR / 2, aggregate=np.mean)
    return oe


def make_envs(sig_d, sig_b):
    sos = ss.butter(4, 150, "low", fs=SR, output="sos")
    kick = ss.sosfiltfilt(sos, sig_d)
    return {"drum": env_of(sig_d), "kick": env_of(kick, fmax=300, n_mels=8), "bass": env_of(sig_b, fmax=400, n_mels=16)}


def calibrate_latency():
    """合成已知時間的 click/kick/bass 脈衝，量測同一包絡偵測器的固定延遲 (s)。"""
    rng = np.random.default_rng(0)
    n = SR * 30
    times = np.arange(1.0, 28.0, 0.5)
    d, b = np.zeros(n), np.zeros(n)
    for t in times:
        i = int(t * SR)
        L = int(0.08 * SR)
        tt = np.arange(L) / SR
        d[i:i + L] += rng.standard_normal(L) * np.exp(-tt * 60) * 0.3 + np.sin(2 * np.pi * 60 * tt) * np.exp(-tt * 25)
        b[i:i + L] += np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 8)
    E = make_envs(d, b)
    out = {}
    for k, e in E.items():
        pk = peaks(e)[0]
        off = [pk[np.argmin(abs(pk - t))] - t for t in times]
        out[k] = float(np.median(off))
    return out


def peaks(e, rel=0.15, dist=0.04):
    thr = rel * np.percentile(e, 99.5)
    p, _ = ss.find_peaks(e, height=thr, distance=max(1, int(dist * SR / HOP)))
    return p * HOP / SR, e[p]


def sample(e, t, lat=0.0, win=0.0):
    """包絡在時間 t 的值（已扣延遲）；win>0 時取 ±win 內最大值。"""
    i = (np.asarray(t) + lat) * SR / HOP
    if win <= 0:
        return e[np.clip(np.round(i).astype(int), 0, len(e) - 1)]
    w = int(round(win * SR / HOP))
    ii = np.clip(np.round(i).astype(int), 0, len(e) - 1)
    return np.array([e[max(0, j - w):j + w + 1].max() for j in ii])


# ---------------------------------------------------------------- Q1
def q1_alignment(beats, envs, lat):
    res = {}
    for k in ("drum", "kick"):
        e = envs[k]
        t, h = peaks(e)
        t = t - lat[k]
        off = np.array([t[np.argmin(abs(t - b))] - b for b in beats])
        # 對照：梳狀相關（整首平均包絡在 beat+δ 的值），找最佳 δ；比 nearest-onset 不受 hi-hat 密度影響
        deltas = np.arange(-0.10, 0.1001, 0.002)
        comb = np.array([sample(e, beats, lat[k] + d).mean() for d in deltas])
        best = deltas[comb.argmax()]
        blocks = []
        for i in range(0, len(beats), 16):
            bb = beats[i:i + 16]
            if len(bb) < 8:
                continue
            c = np.array([sample(e, bb, lat[k] + d).mean() for d in deltas])
            blocks.append({"beats": f"{i}-{i + len(bb) - 1}", "t": round(float(bb[0]), 1), "comb_lag_ms": round(float(deltas[c.argmax()] * 1000), 0)})
        good = abs(off) < 0.04
        slope = np.polyfit(beats[good], off[good] * 1000, 1)[0] * 60 if good.sum() > 10 else None
        # 無 onset 於 80ms 內
        no_on = [int(i) for i, b in enumerate(beats) if np.min(abs(t - b)) > 0.08]
        bad = [int(i) for i in np.where(abs(off) > 0.04)[0]]
        res[k] = {
            "n_onsets": int(len(t)), "median_ms": round(float(np.median(off) * 1000), 1),
            "median_abs_ms": round(float(np.median(abs(off)) * 1000), 1),
            "p90_abs_ms": round(float(np.percentile(abs(off), 90) * 1000), 1),
            "n_gt40": len(bad), "n_no_onset_80ms": len(no_on),
            "comb_best_lag_ms": round(float(best * 1000), 0),
            "comb_value_at_best": round(float(comb.max()), 3), "comb_value_at_0": round(float(comb[len(deltas) // 2]), 3),
            "drift_ms_per_min_(|off|<40)": None if slope is None else round(float(slope), 2),
            "blocks": blocks, "gt40_idx": bad, "no_onset_idx": no_on,
            "gt40_list": [{"i": i, "t": round(float(beats[i]), 2), "off_ms": round(float(off[i] * 1000))} for i in bad],
        }
    return res


# ---------------------------------------------------------------- Q2
def combo_env(envs):
    c = sum(envs[k] / np.percentile(envs[k], 99.5) for k in ("drum", "kick"))
    return c


def grid_score(e, times, lat, shift=0.0):
    return float(sample(e, np.asarray(times) + shift, lat, win=0.025).mean())


def best_uniform(e, lat, a, b, n, jitter=0.06):
    """a→b 之間 n 個等分區間，端點各可 ±jitter 移動；回傳最佳 (score, times)。"""
    best = (-1, None)
    js = np.arange(-jitter, jitter + 1e-9, 0.01)
    for ja in js:
        for jb in js:
            ts = np.linspace(a + ja, b + jb, n + 1)
            s = grid_score(e, ts[1:-1], lat) if n > 1 else 0
            if s > best[0]:
                best = (s, ts)
    return best


def q2_regions(tl_beats, envs, lat):
    e = combo_env(envs)
    bt = np.array([b["t"] for b in tl_beats])
    out = {}
    specs = {
        "bar38_region": dict(ia=145, ib=157, note="anchors = bar37/bar40 downbeats (84.28 / 90.52 s)"),
        "bars49_50_region": dict(ia=193, ib=201, note="anchors = bar49 / bar51 downbeats (110.80 / 115.78 s)"),
    }
    for name, sp in specs.items():
        ia, ib = sp["ia"], sp["ib"]
        a, b = bt[ia], bt[ib]
        n_ai = ib - ia
        latc = (lat["drum"] + lat["kick"]) / 2
        al = {"n_intervals": n_ai, "score": round(grid_score(e, bt[ia + 1:ib], latc), 3), "times": [round(float(x), 2) for x in bt[ia:ib + 1]]}
        cands = {}
        for n in (n_ai - 2, n_ai - 1, n_ai, n_ai + 1):
            s, ts = best_uniform(e, (lat["drum"] + lat["kick"]) / 2, a, b, n)
            cands[n] = {"score": round(float(s), 3), "period_s": round(float((ts[-1] - ts[0]) / n), 3), "times": [round(float(x), 2) for x in ts]}
        # 理論穩定拍距 0.561 對應的區間數
        expect = (b - a) / 0.561
        out[name] = {"note": sp["note"], "span_s": [round(float(a), 2), round(float(b), 2)], "beats_equiv_at_0.561": round(float(expect), 2),
                     "allin1": al, "uniform_candidates": cands, "null_mean_env": round(float(e.mean()), 3)}
    return out


def dp_region(e, lat, lo, hi, tight=400):
    """Ellis DP 拍追蹤（librosa.beat.beat_track）跑在 [lo,hi] 秒的 drum+kick 包絡上，起始速度 107 bpm。"""
    fps = SR / HOP
    i0, i1 = int(lo * fps), int(hi * fps)
    _, bf = librosa.beat.beat_track(onset_envelope=e[i0:i1], sr=SR, hop_length=HOP, start_bpm=107, tightness=tight, trim=False)
    return (bf + i0) * HOP / SR - lat


def q2_corrections(tl_beats, envs, lat):
    """針對兩個不規則區，比較 allin1 內部拍與（DP / 等距）修正拍在 drum+kick 包絡上的吻合度，並產生修正建議。"""
    e = combo_env(envs)
    latc = (lat["drum"] + lat["kick"]) / 2
    bt = np.array([b["t"] for b in tl_beats])
    regions = {
        # name: (first allin1 beat idx to replace, last idx to replace, left anchor idx, right anchor idx, DP window)
        "bar38_region": dict(rep=(149, 152), anc=(148, 153), win=(76, 100)),
        "bars49_50_region": dict(rep=(196, 196), anc=(195, 197), win=(103, 124)),
    }
    out = {}
    for name, r in regions.items():
        ia, ib = r["anc"]
        ri, rj = r["rep"]
        a, b = bt[ia], bt[ib]
        dp = dp_region(e, latc, *r["win"])
        dp_in = [float(x) for x in dp if a + 0.2 < x < b - 0.2]
        # 等距版：把 DP 找到的區間頭尾之間等分
        ends = [float(x) for x in dp if a - 0.1 < x < b + 0.1]
        n_in = len(dp_in)
        uni_all = np.linspace(a if name != "bar38_region" else dp_in[0], b, n_in + 1 if name == "bar38_region" else n_in + 2)
        uni_in = [float(x) for x in uni_all[1:-1]] if name != "bar38_region" else [float(x) for x in uni_all[:-1]]
        al_in = [float(x) for x in bt[ri:rj + 1]]
        sc = lambda ts: round(grid_score(e, ts, latc), 3)
        out[name] = {
            "anchors_t": [round(float(a), 2), round(float(b), 2)],
            "allin1_interior": {"t": [round(x, 2) for x in al_in], "n": len(al_in), "score": sc(al_in)},
            "dp_interior": {"t": [round(x, 2) for x in dp_in], "n": len(dp_in), "score": sc(dp_in)},
            "uniform_interior": {"t": [round(x, 2) for x in uni_in], "n": len(uni_in), "score": sc(uni_in)},
            "dp_intervals_s": [round(float(x), 3) for x in np.diff([a] + dp_in + [b])],
            "null_mean_env": round(float(e.mean()), 3),
        }
    return out


# ---------------------------------------------------------------- Q3 phase
def band_env(x, lo, hi):
    sos = ss.butter(4, [lo, hi], "band", fs=SR, output="sos")
    return env_of(ss.sosfiltfilt(sos, x), n_mels=16, fmax=hi)


def vocal_phrase_starts(voc, gap=0.4):
    r = librosa.feature.rms(y=voc, frame_length=1024, hop_length=HOP)[0]
    r = np.convolve(r, np.ones(9) / 9, "same")
    on = r > 0.15 * np.percentile(r, 95)
    starts, last_on = [], -9
    for i in range(len(on)):
        if on[i]:
            if i * HOP / SR - last_on > gap:
                starts.append(i * HOP / SR)
            last_on = i * HOP / SR
    return np.array(starts)


def beat_sync_chroma(sig_mix, bt):
    ch = librosa.feature.chroma_cqt(y=sig_mix, sr=SR, hop_length=512)
    ft = librosa.frames_to_time(np.arange(ch.shape[1]), sr=SR, hop_length=512)
    out = []
    edges = np.append(bt, bt[-1] + 0.561)
    for k in range(len(bt)):
        m = (ft >= edges[k]) & (ft < edges[k + 1])
        out.append(ch[:, m].mean(1) if m.any() else np.zeros(12))
    return np.array(out)


def cos_dist(a, b):
    return 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def q3_phase(tl, envs, sigs, lat, dr_band_env, excl):
    beats = tl["beats"]
    bt = np.array([b["t"] for b in beats])
    cls = np.array([b["beat_in_bar"] for b in beats])
    # 非 4 拍的小節（拍點修正後的 bar 38/49、曲尾）不參與相位統計
    odd = {b["bar"] for b in tl["meta"].get("irregular_bars", [])}
    ok = np.array([i for i in range(len(bt)) if i >= 1 and beats[i]["bar"] not in odd
                   and not any(lo <= i <= hi for lo, hi in excl)])
    res = {"per_class": {}, "phase_scores": {}}
    # --- 逐拍位置平均（以全曲平均為 1）
    cues = {}
    cues["kick (drums<150Hz onset)"] = sample(envs["kick"], bt, lat["kick"] + 0.015, win=0.03)
    cues["bass onset"] = sample(envs["bass"], bt, lat["bass"] + 0.015, win=0.03)
    cues["snare (drums 1.5-6kHz onset)"] = sample(dr_band_env, bt, 0.015, win=0.03)
    # 和聲變化：beat-sync chroma（bass+other）與前一拍的距離
    C = sigs["chroma"]
    cues["harmonic change (bass+other chroma)"] = np.array([0] + [cos_dist(C[k], C[k - 1]) for k in range(1, len(bt))])
    # 一小節內「與前一小節同位置」的 chroma 差：以 4 拍窗平均比較，避免單拍雜訊
    cues["harmonic change 4-beat window"] = np.array([cos_dist(C[max(0, k - 4):k].mean(0) if k >= 1 else C[0], C[k:k + 4].mean(0)) if k >= 4 else 0 for k in range(len(bt))])
    for name, v in cues.items():
        m = {int(c): float(v[ok][cls[ok] == c].mean() / v[ok].mean()) for c in (1, 2, 3, 4)}
        res["per_class"][name] = {str(c): round(m[c], 3) for c in m}
    # --- 事件類 cue：落在各 beat_in_bar 的比例
    def cls_of_next(t, back=0.10, maxd=0.45):
        j = np.searchsorted(bt, t - back)
        if j >= len(bt) or bt[j] - t > maxd:
            return None
        return int(cls[j])
    def hist(ts):
        h = {c: 0 for c in (1, 2, 3, 4)}
        for t in ts:
            c = cls_of_next(t)
            if c in h:
                h[c] += 1
        n = sum(h.values())
        return {str(c): (h[c], round(h[c] / max(n, 1), 3)) for c in h}, n
    srt_starts = [l["t_start"] for l in tl["lyrics"]]
    res["event_hist"] = {}
    h, n = hist(srt_starts); res["event_hist"]["SRT cue starts (next beat >= start-0.1s, within 0.45s)"] = {"hist": h, "n": n}
    def hist_nearest(ts, maxd=0.2):
        h = {c: 0 for c in (1, 2, 3, 4)}
        for t in ts:
            j = int(np.argmin(abs(bt - t)))
            if abs(bt[j] - t) <= maxd and int(cls[j]) in h:
                h[int(cls[j])] += 1
        n = sum(h.values())
        return {str(c): (h[c], round(h[c] / max(n, 1), 3)) for c in h}, n
    h, n = hist_nearest(srt_starts); res["event_hist"]["SRT nearest beat (+-0.2s)"] = {"hist": h, "n": n}
    vs = sigs["vocal_starts"]
    h, n = hist(vs); res["event_hist"]["vocal-stem phrase starts (gap>0.4s)"] = {"hist": h, "n": n, "times": [round(float(x), 2) for x in vs]}
    # 結構新奇度：以 beat-sync 特徵 [chroma(12)+rms(stems)]，比較前 4 拍與後 4 拍平均，取峰值
    F = sigs["struct_feat"]
    nov = np.zeros(len(bt))
    for k in range(4, len(bt) - 4):
        nov[k] = np.linalg.norm(F[k - 4:k].mean(0) - F[k:k + 4].mean(0))
    pk, _ = ss.find_peaks(nov, distance=6, height=np.percentile(nov, 80))
    ht = {c: 0 for c in (1, 2, 3, 4)}
    pk = [k for k in pk if int(cls[k]) in ht]
    for k in pk:
        ht[int(cls[k])] += 1
    res["event_hist"]["structural novelty peaks (beat idx, 4-vs-4 beat)"] = {"hist": {str(c): (ht[c], round(ht[c] / max(len(pk), 1), 3)) for c in ht}, "n": int(len(pk)),
        "times": [round(float(bt[k]), 2) for k in pk]}
    res["novelty_curve"] = nov
    # 段落邊界（sections.json 主邊界）相對 allin1 拍位：改以邊界附近(±1 beat)結構新奇度最高的拍位來看
    sec_ts = [s["t_start"] for s in tl["sections"][1:]]
    ks = [int(np.argmin(abs(bt - t))) for t in sec_ts]
    res["section_boundaries_nearest_beat_cls"] = [{"t": round(t, 2), "beat_in_bar": int(cls[k])} for t, k in zip(sec_ts, ks)]
    # 以 novelty 在每個段落邊界 ±2 拍內的峰值落在哪個 beat_in_bar
    near = {c: 0 for c in (1, 2, 3, 4)}
    detail = []
    for t, k in zip(sec_ts, ks):
        w = range(max(4, k - 2), min(len(bt) - 4, k + 3))
        kk = max(w, key=lambda q: nov[q])
        if int(cls[kk]) in near:
            near[int(cls[kk])] += 1
        detail.append({"section_t": round(t, 2), "nov_peak_beat_t": round(float(bt[kk]), 2), "beat_in_bar": int(cls[kk])})
    res["event_hist"]["novelty peak within +-2 beats of each section boundary"] = {"hist": {str(c): (near[c], round(near[c] / len(ks), 3)) for c in near}, "n": len(ks), "detail": detail}
    # --- 4 種相位的總表：phase h = 「beat_in_bar==h 的拍才是 1 拍」
    for h_ in (1, 2, 3, 4):
        row = {}
        for name in ("kick (drums<150Hz onset)", "bass onset", "harmonic change (bass+other chroma)", "harmonic change 4-beat window"):
            row[name] = res["per_class"][name][str(h_)]
        # 後拍：snare 在 h+1、h+3 vs h、h+2
        sn = res["per_class"]["snare (drums 1.5-6kHz onset)"]
        on = [str((h_ - 1 + k) % 4 + 1) for k in (1, 3)]
        off = [str((h_ - 1 + k) % 4 + 1) for k in (0, 2)]
        row["backbeat (snare on beats 2&4 minus 1&3)"] = round(float(np.mean([sn[c] for c in on]) - np.mean([sn[c] for c in off])), 3)
        for name, d in res["event_hist"].items():
            row[name.split(" (")[0]] = d["hist"][str(h_)][1]
        res["phase_scores"][f"phase_{h_}"] = row
    return res


def q3_pickup(sigs_raw, envs, lat, first_beat=2.22, period=0.561):
    """0 ~ first_beat 之間：各 stem 的 RMS、drum onset，及向前延伸的拍點上的包絡值（對照全曲平均）。"""
    out = {"rms_per_stem_0.0-2.22": {}, "rms_per_stem_3-6s_ref": {}}
    for k, x in sigs_raw.items():
        r = librosa.feature.rms(y=x, frame_length=1024, hop_length=HOP)[0]
        out["rms_per_stem_0.0-2.22"][k] = round(float(r[:int(first_beat * SR / HOP)].mean()), 4)
        out["rms_per_stem_3-6s_ref"][k] = round(float(r[int(3 * SR / HOP):int(6 * SR / HOP)].mean()), 4)
    ext = [round(first_beat - period * k, 2) for k in (1, 2, 3, 4) if first_beat - period * k > 0.2]
    out["backward_beats"] = ext
    for k in ("drum", "kick", "bass"):
        t, h = peaks(envs[k]); t = t - lat[k]
        near = [(round(float(t[np.argmin(abs(t - b))]), 2), round(float(t[np.argmin(abs(t - b))] - b) * 1000)) for b in ext]
        out[f"{k}_nearest_peak(t, off_ms)"] = near
        out[f"{k}_env_at_backward_beats"] = [round(float(sample(envs[k], np.array([b]), lat[k] + 0.015, win=0.03)[0]), 2) for b in ext]
        out[f"{k}_env_mean_song"] = round(float(envs[k].mean()), 2)
    prof = {}
    for k, x in sigs_raw.items():
        r = librosa.feature.rms(y=x, frame_length=2048, hop_length=HOP)[0]
        ref = np.percentile(r, 95)
        prof[k] = [round(float(10 * r[int(a * SR / HOP):int((a + 0.28) * SR / HOP)].mean() / ref), 1) for a in np.arange(0, 3.4, 0.28)]
    out["rms_profile_x10_of_p95_per_0.28s_from_0s"] = prof
    # 0-2.22 內的 onset 時間
    for k in ("drum", "bass"):
        t, h = peaks(envs[k]); t = t - lat[k]
        m = t < first_beat + 0.05
        out[f"{k}_peaks_before_first_beat"] = [(round(float(a), 2), round(float(b), 1)) for a, b in zip(t[m], h[m])]
    return out


# ---------------------------------------------------------------- Q4 per-bar
def q4_bars(tl, envs, rms, bar_chroma_change, nov, bt, bar_ranges):
    def rmsm(k, t0, t1):
        return float(rms[k][int(t0 * SR / HOP):max(int(t1 * SR / HOP), int(t0 * SR / HOP) + 1)].mean())
    ref = {k: float(np.percentile(v, 95)) for k, v in rms.items()}
    td, _ = peaks(envs["drum"], rel=0.3)
    tk, _ = peaks(envs["kick"], rel=0.3)
    tb, _ = peaks(envs["bass"], rel=0.3)
    bars = {b["bar"]: b for b in tl["bars"]}
    rows = {}
    for lo, hi, label in bar_ranges:
        tab = []
        for n in range(lo, hi + 1):
            b = bars.get(n)
            if not b:
                continue
            t0, t1 = b["t_start"], b["t_end"]
            beat_len = (t1 - t0) / 4
            cnt = lambda ts, a, c: int(((ts >= a) & (ts < c)).sum())
            last = cnt(td, t1 - beat_len, t1)
            tab.append({
                "bar": n, "t": round(t0, 2),
                "drum": round(10 * rmsm("drums", t0, t1) / ref["drums"], 1), "kick": round(10 * rmsm("kick", t0, t1) / ref["kick"], 1),
                "hat": round(10 * rmsm("hat", t0, t1) / ref["hat"], 1), "bass": round(10 * rmsm("bass", t0, t1) / ref["bass"], 1),
                "other": round(10 * rmsm("other", t0, t1) / ref["other"], 1), "vocal": round(10 * rmsm("vocals", t0, t1) / ref["vocals"], 1),
                "drum_onsets": cnt(td, t0, t1), "kick_onsets": cnt(tk, t0, t1), "bass_onsets": cnt(tb, t0, t1),
                "last_beat_drum_onsets": last,
                "harm_change_at_downbeat": round(float(bar_chroma_change.get(n, 0)), 3),
                "novelty_at_downbeat": round(float(nov[int(np.argmin(abs(bt - t0)))]), 2),
            })
        rows[label] = tab
    return rows


# ---------------------------------------------------------------- outputs
def make_clicks(y, sr, beats, secs, g=0.5):
    down, other = burst(sr, 1800, 0.06, g), burst(sr, 900, 0.04, g * 0.45)
    for b in beats:
        add(y, b["t"], down if b["is_downbeat"] else other, sr)
    mark, pip = burst(sr, 330, 0.35, g * 1.2, decay=6.0), burst(sr, 1200, 0.07, g, decay=25.0)
    for s in secs[1:]:
        add(y, s["t_start"], mark, sr)
    for s in secs:
        if "alt_t" in s:
            add(y, s["alt_t"], pip, sr); add(y, s["alt_t"] + 0.12, pip, sr)


def write_excerpt(audio, path, t0, t1, beats, secs):
    y, sr = sf.read(audio, dtype="float32", always_2d=True)
    y = y * 0.8
    make_clicks(y, sr, beats, secs)
    seg = y[int(t0 * sr):int(t1 * sr)].copy()
    peak = float(np.abs(seg).max())
    if peak > 0.99:
        seg *= 0.99 / peak
    sf.write(path, seg, sr, subtype="PCM_16")


def corrected_beats(tl_beats, corr):
    """依 beat_corrections 的 replace 規則產生修正後的 beats（只給 click 用）。"""
    out = [dict(b) for b in tl_beats]
    for c in corr["corrections"]:
        lo, hi = c["replace_allin1_beats_t"]
        out = [b for b in out if not (lo - 1e-6 <= b["t"] <= hi + 1e-6)]
        for t, d in zip(c["new_beats_t"], c["new_is_downbeat"]):
            out.append({"t": t, "is_downbeat": d})
    return sorted(out, key=lambda b: b["t"])


def plot_spot(path, title, t0, t1, envs, lat, tl, alt_beats=None, extra_lines=None):
    fig, ax = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    tt = np.arange(len(envs["drum"])) * HOP / SR
    m = (tt >= t0 - 0.2) & (tt <= t1 + 0.2)
    def norm(e): return e[m] / np.percentile(e, 99.5)
    ax[0].plot(tt[m] - lat["drum"], norm(envs["drum"]), color="#2b6cb0", lw=1, label="drum-stem onset")
    ax[0].plot(tt[m] - lat["kick"], norm(envs["kick"]), color="#c53030", lw=1.2, label="kick band (<150Hz) onset")
    ax[1].plot(tt[m] - lat["bass"], norm(envs["bass"]), color="#2f855a", lw=1.2, label="bass-stem onset")
    bs = [b for b in tl["beats"] if t0 - 0.2 <= b["t"] <= t1 + 0.2]
    for a in ax:
        for b in bs:
            a.axvline(b["t"], color="k", lw=2.2 if b["is_downbeat"] else 0.7, alpha=0.75 if b["is_downbeat"] else 0.45)
        for s in tl["sections"][1:]:
            if t0 <= s["t_start"] <= t1:
                a.axvline(s["t_start"], color="#6b46c1", lw=1.5, ls="--")
        for s in tl["sections"]:
            if "alt_t" in s and t0 <= s["alt_t"] <= t1:
                a.axvline(s["alt_t"], color="#dd6b20", lw=1.5, ls=":")
        for x in (extra_lines or []):
            if t0 <= x <= t1:
                a.axvline(x, color="#dd6b20", lw=1.2, ls=":")
    top = ax[0].get_ylim()[1]
    for b in bs:
        if t0 <= b["t"] <= t1:
            lab = f"bar {b['bar']}" if b["is_downbeat"] else str(b["beat_in_bar"])
            ax[0].text(b["t"], top * 1.02, lab, ha="center", va="bottom", fontsize=9 if b["is_downbeat"] else 8, fontweight="bold" if b["is_downbeat"] else "normal")
    for s in tl["sections"][1:]:
        if t0 <= s["t_start"] <= t1:
            ax[1].text(s["t_start"], 0.98, " " + s["label"], transform=ax[1].get_xaxis_transform(), color="#6b46c1", va="top", fontsize=9)
    for s in tl["sections"]:
        if "alt_t" in s and t0 <= s["alt_t"] <= t1:
            ax[1].text(s["alt_t"], 0.80, f" alt {s['label']} (bar {s['alt_bar']})", transform=ax[1].get_xaxis_transform(), color="#dd6b20", va="top", fontsize=9)
    if alt_beats:
        for x in alt_beats:
            if t0 <= x <= t1:
                ax[1].plot([x], [-0.06], marker="^", color="#2f855a", ms=9, clip_on=False, transform=ax[1].get_xaxis_transform())
        ax[1].text(0.005, -0.28, "green triangles = proposed corrected beats", transform=ax[1].transAxes, color="#2f855a", fontsize=9)
    ax[0].set_xlim(t0, t1)
    ax[0].legend(loc="upper right", fontsize=8); ax[1].legend(loc="upper right", fontsize=8)
    ax[0].set_title(title + "   (thin=beat, thick=downbeat(allin1), dashed=section, dotted=alt)", fontsize=11, pad=22)
    ax[1].set_xlabel("seconds")
    for a in ax:
        a.set_ylim(bottom=0); a.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default="audio/song.wav")
    ap.add_argument("--timeline", default="analysis/timeline.json")
    ap.add_argument("--out", default="analysis/check")
    ap.add_argument("--stems", default=None, help="demucs stems 目錄（含 song/drums.wav…），預設 <out>/stems")
    ap.add_argument("--no-media", action="store_true", help="只算數字，不輸出 excerpts / plots")
    args = ap.parse_args()

    out = Path(args.out)
    stem_dir = ensure_stems(Path(args.audio), Path(args.stems) if args.stems else out / "stems")
    tl = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    bt = np.array([b["t"] for b in tl["beats"]])
    sig = {n: librosa.load(stem_dir / f"{n}.wav", sr=SR)[0] for n in ("drums", "bass", "other", "vocals")}
    lat = calibrate_latency()
    print("detector latency (s):", {k: round(v, 4) for k, v in lat.items()})
    envs = make_envs(sig["drums"], sig["bass"])
    excl = [(145, 157), (193, 201)]
    report = {"latency_s": lat}

    # Q1
    report["Q1"] = q1_alignment(bt, envs, lat)
    for k, r in report["Q1"].items():
        print(f"Q1 {k}: median {r['median_ms']} ms, p90|off| {r['p90_abs_ms']} ms, >40ms: {r['n_gt40']}, no onset<=80ms: {r['n_no_onset_80ms']}, comb best lag {r['comb_best_lag_ms']} ms, drift {r['drift_ms_per_min_(|off|<40)']} ms/min")

    # Q2
    report["Q2"] = q2_corrections(tl["beats"], envs, lat)
    report["Q2_uniform_interval_test"] = q2_regions(tl["beats"], envs, lat)
    for k, r in report["Q2"].items():
        print("Q2", k, {n: (r[n]["n"], r[n]["score"]) for n in ("allin1_interior", "dp_interior", "uniform_interior")}, "null", r["null_mean_env"])
    r1, r2 = report["Q2"]["bar38_region"], report["Q2"]["bars49_50_region"]
    corr = {"note": "Proposed corrections (NOT applied to timeline.json). Replace allin1 beats whose t lies in replace_allin1_beats_t with new_beats_t. Region 1 turns bar 38 into a 3-beat bar; region 2 turns bar 49 into a 5-beat bar. Beat count stays 232.",
            "corrections": [
                {"region": "bar38", "replace_allin1_beats_t": [86.41, 87.75], "allin1_t": r1["allin1_interior"]["t"],
                 "new_beats_t": r1["uniform_interior"]["t"], "new_is_downbeat": [True, False, False],
                 "dp_alternative_t": r1["dp_interior"]["t"], "bar38_beats": 3,
                 "evidence": {"score_allin1": r1["allin1_interior"]["score"], "score_new": r1["uniform_interior"]["score"], "score_dp": r1["dp_interior"]["score"]}},
                {"region": "bar49_50", "replace_allin1_beats_t": [112.72, 112.72], "allin1_t": r2["allin1_interior"]["t"],
                 "new_beats_t": r2["dp_interior"]["t"], "new_is_downbeat": [False, False],
                 "uniform_alternative_t": r2["uniform_interior"]["t"], "bar49_beats": 5,
                 "evidence": {"score_allin1": r2["allin1_interior"]["score"], "score_new": r2["dp_interior"]["score"], "score_uniform": r2["uniform_interior"]["score"]}},
            ]}
    (out).mkdir(parents=True, exist_ok=True)
    (out / "beat_corrections.json").write_text(json.dumps(jsonable(corr), ensure_ascii=False, indent=2), encoding="utf-8")

    # Q3
    mix_ho = sig["bass"] + sig["other"]
    chroma = beat_sync_chroma(mix_ho, bt)
    rms = {k: librosa.feature.rms(y=v, frame_length=1024, hop_length=HOP)[0] for k, v in sig.items()}
    sos = ss.butter(4, 150, "low", fs=SR, output="sos")
    sosh = ss.butter(4, 5000, "high", fs=SR, output="sos")
    rms["kick"] = librosa.feature.rms(y=ss.sosfiltfilt(sos, sig["drums"]), frame_length=1024, hop_length=HOP)[0]
    rms["hat"] = librosa.feature.rms(y=ss.sosfiltfilt(sosh, sig["drums"]), frame_length=1024, hop_length=HOP)[0]
    # 結構特徵：chroma + 各 stem 的拍內 log-RMS
    feats = [chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9)]
    for k in ("drums", "bass", "other", "vocals"):
        v = np.array([rms[k][int(a * SR / HOP):max(int(a * SR / HOP) + 1, int((a + 0.561) * SR / HOP))].mean() for a in bt])
        feats.append(np.log1p(v / np.percentile(rms[k], 95) * 10)[:, None])
    F = np.hstack(feats)
    snare = band_env(sig["drums"], 1500, 6000)
    sigs = {"chroma": chroma, "struct_feat": F, "vocal_starts": vocal_phrase_starts(sig["vocals"])}
    q3 = q3_phase(tl, envs, sigs, lat, snare, excl)
    nov = q3.pop("novelty_curve")
    q3["pickup"] = q3_pickup({k: sig[k] for k in sig}, envs, lat)
    report["Q3"] = q3
    print("Q3 per-class (mean/overall), classes 1..4 = beat_in_bar:")
    for k, v in q3["per_class"].items():
        print("  ", k, v)
    print("Q3 backbeat/event cues by phase (phase h = beat_in_bar==h is the downbeat):")
    for k, v in q3["phase_scores"].items():
        print("  ", k, v)
    for k, v in q3["event_hist"].items():
        print("  ", k, v["hist"], "n", v["n"])
    print("Q3 pickup:", json.dumps(jsonable(q3["pickup"]), ensure_ascii=False))

    # Q4
    cc = {}
    bars = tl["bars"]
    for i, b in enumerate(bars):
        k = int(np.argmin(abs(bt - b["t_start"])))
        if k >= 4 and k + 4 <= len(bt):
            cc[b["bar"]] = cos_dist(chroma[k - 4:k].mean(0), chroma[k:k + 4].mean(0))
    ranges = [(14, 19, "prechorus bar17 vs alt bar16"), (19, 24, "chorus1 bar21 vs alt bar22"), (35, 40, "build bar37 vs alt bar38"), (54, 58, "outro bar57 vs alt bar58")]
    report["Q4"] = q4_bars(tl, envs, rms, cc, nov, bt, ranges)
    for label, tab in report["Q4"].items():
        print("Q4", label)
        print("   bar    t  drum kick  hat bass other vocal | d_on k_on b_on last_beat_d | harmchg  novelty")
        for r in tab:
            print(f"   {r['bar']:>3} {r['t']:6.2f} {r['drum']:5} {r['kick']:4} {r['hat']:4} {r['bass']:4} {r['other']:5} {r['vocal']:5} | {r['drum_onsets']:>3} {r['kick_onsets']:>3} {r['bass_onsets']:>3} {r['last_beat_drum_onsets']:>5}       | {r['harm_change_at_downbeat']:.3f}   {r['novelty_at_downbeat']:.2f}")

    (out / "verify_report.json").write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=1), encoding="utf-8")

    if args.no_media:
        return
    # Q5 media
    exd, pld = out / "excerpts", out / "plots"
    exd.mkdir(exist_ok=True); pld.mkdir(exist_ok=True)
    for d in (exd, pld):
        for f in d.glob("[0-9][0-9]_*"):
            f.unlink()
    cb = corrected_beats(tl["beats"], corr)
    for b in cb:  # 給修正版 click 補 bar 標記不需要；只需 t / is_downbeat
        b.setdefault("is_downbeat", False)
    cor_t = [t for c in corr["corrections"] for t in c["new_beats_t"]]
    sec = {s["label"]: s for s in tl["sections"]}
    spots = [
        ("01_first_downbeat_0-8s", 0, 8, False),
        ("02_prechorus_bar17_vs_alt16", sec["prechorus"]["t_start"] - 5, sec["prechorus"]["t_start"] + 5, False),
        ("03_chorus1_bar21_vs_alt22", sec["chorus1"]["t_start"] - 5, sec["chorus1"]["alt_t"] + 3, False),
        ("04_build_bar37_vs_alt38", 82, 92, False),
        ("05_bar38_irregular_allin1_clicks", 83, 93, False),
        ("06_bar38_irregular_CORRECTED_clicks", 83, 93, True),
        ("07_bars49-50_irregular_allin1_clicks", 108, 118, False),
        ("08_bars49-50_irregular_CORRECTED_clicks", 108, 118, True),
        ("09_outro_bar57_vs_alt58", 125.5, 134.2, False),
    ]
    for name, t0, t1, corrected in spots:
        beats_for_click = cb if corrected else tl["beats"]
        write_excerpt(args.audio, exd / f"{name}.wav", t0, t1, beats_for_click, tl["sections"])
        if not corrected:
            plot_spot(pld / f"{name}.png", name, t0, t1, envs, lat, tl,
                      alt_beats=cor_t if "irregular" in name else None)
        print("wrote", name)
    print("report ->", out / "verify_report.json")


if __name__ == "__main__":
    sys.exit(main())
