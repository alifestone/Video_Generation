#!/usr/bin/env python3
"""P1：合併 allin1、librosa 與 SRT，產出 analysis/timeline.json（只用標準函式庫）。

正式 P1 指令（canonical）：
python3 tools/build_timeline.py --allin1 analysis/struct/song.json --energy analysis/librosa.json --srt audio/lyrics.srt --audio audio/song.wav --sections analysis/sections.json --beat-corrections analysis/beat_corrections.json -o analysis/timeline.json

其他選項：--bpm-mode auto|as-is|double|halve（拍點修正與 hits 以 as-is 的 allin1 拍點為前提）。
--beat-corrections：以 beat_corrections.json 替換 allin1 的拍點（見 apply_beat_corrections）。
--sections 檔內的頂層 `hits` 會輸出為 timeline.hits，並在對應小節標 bars[i].hit。

規則：幀號 = round(t * 24)；frame_end 不含；小節從第一個首拍起算，之前的拍點屬於 bar 0（pickup）。
"""
import argparse, bisect, hashlib, json, math, re, statistics, sys
from pathlib import Path

FPS = 24
BEATS_PER_BAR = 4


def frame(t):
    return int(round(t * FPS))


def r3(x):
    return round(float(x), 3)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- SRT
_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def parse_srt(text):
    """回傳 [{"id","t_start","t_end","text"}]；多行文字以空白連接。"""
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").lstrip("﻿").strip()):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        for k, line in enumerate(lines):
            m = _TS.search(line)
            if not m:
                continue
            g = m.groups()
            cid = int(lines[k - 1]) if k > 0 and lines[k - 1].isdigit() else len(cues) + 1
            cues.append({"id": cid, "t_start": _sec(*g[:4]), "t_end": _sec(*g[4:]),
                         "text": " ".join(lines[k + 1:])})
            break
    return cues


# ---------------------------------------------------------------- beats
def beats_from_allin1(a, warnings):
    """回傳 (times, positions)，positions 取自 beat_positions（長度不符時退回用 downbeats 推算）。"""
    times = [float(t) for t in a["beats"]]
    pos = a.get("beat_positions")
    if pos and len(pos) == len(times):
        return times, [int(p) for p in pos]
    warnings.append("allin1 beat_positions missing or length mismatch; derived from downbeats")
    med = statistics.median([b - a_ for a_, b in zip(times, times[1:])]) if len(times) > 1 else 1
    downs = [float(d) for d in a.get("downbeats", [])]
    pos, cur = [], BEATS_PER_BAR
    for t in times:
        if any(abs(t - d) < med / 2 for d in downs):
            cur = 1
        else:
            cur = cur % BEATS_PER_BAR + 1
        pos.append(cur)
    return times, pos


def transform_grid(times, pos, mode):
    """double：在拍點間插入中點；halve：隔拍保留（首個首拍保持為拍點）。
    首拍之後每 4 拍一小節，由第一個首拍的位置重新計算 beat_in_bar。"""
    if mode in ("auto", "as-is"):
        return times, pos
    i0 = next((i for i, p in enumerate(pos) if p == 1), 0)
    if mode == "double":
        new = []
        for i, t in enumerate(times):
            new.append(t)
            if i + 1 < len(times):
                new.append((t + times[i + 1]) / 2)
        anchor = 2 * i0
    else:  # halve
        keep = [i for i in range(len(times)) if (i - i0) % 2 == 0]
        new = [times[i] for i in keep]
        anchor = keep.index(i0)
    return new, [((j - anchor) % BEATS_PER_BAR) + 1 for j in range(len(new))]


def apply_beat_corrections(path, times, pos, tol=0.005):
    """依 beat_corrections.json 替換拍點；回傳 (times, pos, corrected_flags, region_names)。
    每個 region：移除 allin1_t 中每個拍點（±5ms，必須各自對到一個），插入 new_beats_t（new_is_downbeat 標首拍）。
    之後從每個首拍起重算 beat_in_bar（首個首拍之前的 pickup 拍維持原值）。"""
    items = [[t, p, False, False] for t, p in zip(times, pos)]  # t, pos, corrected, forced downbeat
    regions = []
    for c in load_json(path)["corrections"]:
        name = c["region"]
        new_t, new_d = c["new_beats_t"], c["new_is_downbeat"]
        if len(new_t) != len(new_d):
            raise SystemExit(f"{path}: region {name}: new_beats_t / new_is_downbeat length mismatch")
        for old in c["allin1_t"]:
            hits = [k for k, it in enumerate(items) if not it[2] and abs(it[0] - old) <= tol]
            if len(hits) != 1:
                raise SystemExit(f"{path}: region {name}: allin1_t {old} matched {len(hits)} beats (need exactly 1)")
            items[hits[0]] = None
            items = [it for it in items if it is not None]
        for t, d in zip(new_t, new_d):
            items.append([float(t), 1 if d else 0, True, bool(d)])
        items.sort(key=lambda it: it[0])
        regions.append(name)
    out_t, out_p, flags = [], [], []
    cur = None  # None = 尚未遇到首拍
    for t, p, corr, forced in items:
        if p == 1:
            cur = 1
        elif cur is not None:
            cur += 1
        out_t.append(t)
        out_p.append(1 if p == 1 else (cur if cur is not None else p))
        flags.append(corr)
    return out_t, out_p, flags, regions


def assign_bars(pos):
    bars, bar = [], 0
    for p in pos:
        if p == 1:
            bar += 1
        bars.append(bar)
    return bars


def decide_bpm(a_bpm, lib, mode, warnings):
    lib_bpm = lib.get("librosa_bpm")
    cands = lib.get("tempo_candidates", [])
    if mode == "double":
        bpm, src = a_bpm * 2, "allin1_doubled"
    elif mode == "halve":
        bpm, src = a_bpm / 2, "allin1_halved"
    else:
        bpm, src = a_bpm, "allin1"
        if mode == "auto" and lib_bpm:
            ratio = lib_bpm / a_bpm
            if abs(ratio - 2) < 0.08 * 2 or abs(ratio - 0.5) < 0.08 * 0.5:
                warnings.append(f"allin1 bpm {a_bpm:g} and librosa bpm {lib_bpm:g} differ by ~2x; "
                                f"kept allin1 (confirm {a_bpm:g} vs {2*a_bpm:g} / {a_bpm/2:g} by ear; "
                                f"use --bpm-mode double|halve to change)")
    return bpm, src, {"allin1": a_bpm, "librosa": lib_bpm, "librosa_candidates": cands}


# ---------------------------------------------------------------- sections
def build_sections(segments, downbeat_ts, duration, beat_sec, warnings):
    segs = sorted(({"label": s["label"], "start": float(s["start"]), "end": float(s["end"])}
                   for s in segments), key=lambda s: s["start"])
    kept = []
    for s in segs:
        if s["label"] in ("start", "end"):
            warnings.append(f"dropped allin1 '{s['label']}' segment "
                            f"[{s['start']:.3f}, {s['end']:.3f}], merged into neighbour")
            continue
        if s["end"] - s["start"] <= 1e-6:
            warnings.append(f"dropped zero-length '{s['label']}' segment at {s['start']:.3f}")
            continue
        kept.append(s)
    if not kept:
        raise SystemExit("allin1 has no usable segments")
    # 邊界 = 各段起點；首段拉到 0，末段拉到 duration，間隙/重疊由前一段吸收
    for a, b in zip(kept, kept[1:]):
        if abs(a["end"] - b["start"]) > 0.05:
            warnings.append(f"gap/overlap between '{a['label']}' end {a['end']:.3f} and "
                            f"'{b['label']}' start {b['start']:.3f}; boundary taken at next start")
    if kept[0]["start"] > 0.05:
        warnings.append(f"first segment starts at {kept[0]['start']:.3f}; extended to 0")
    raw = [0.0] + [s["start"] for s in kept[1:]] + [duration]
    counts = {}
    for s in kept:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    seen, labels = {}, []
    for s in kept:
        seen[s["label"]] = seen.get(s["label"], 0) + 1
        labels.append(s["label"] + str(seen[s["label"]]) if counts[s["label"]] > 1 else s["label"])
    # 內部邊界吸附到最近的首拍
    snapped, snap_beats = [0.0], [0.0]
    for k in range(1, len(raw) - 1):
        t = raw[k]
        best = min(downbeat_ts, key=lambda d: abs(d - t)) if downbeat_ts else t
        if best <= snapped[-1] + 1e-6 or best >= duration:
            warnings.append(f"boundary {t:.3f} could not snap to a downbeat without collapsing a "
                            f"section; kept raw time")
            best = t
        snapped.append(best)
        snap_beats.append(round((best - t) / beat_sec, 2))
    snapped.append(duration)
    for k in range(1, len(snapped)):
        if frame(snapped[k]) <= frame(snapped[k - 1]):
            warnings.append(f"section {labels[k-1]} is < 1 frame after snapping")
    secs = []
    for k, label in enumerate(labels):
        secs.append({"idx": k, "label": label, "t_start": snapped[k], "t_end": snapped[k + 1],
                     "t_start_raw": raw[k], "t_end_raw": raw[k + 1], "snap_beats": snap_beats[k]})
    return secs


def override_sections(path, down_t, duration):
    """讀手動段落：{"sections":[{"label","bar",...}]}，每段從該小節首拍開始（bar 0 = 0 秒）。"""
    items = load_json(path)["sections"]
    n_bars = len(down_t) - 1
    prev = None
    for it in items:
        b = it["bar"]
        if not isinstance(b, int) or b < 0 or b > n_bars or (prev is None and b != 0) or (prev is not None and b <= prev):
            raise SystemExit(f"{path}: bad bar {b!r} for '{it.get('label')}' (need strictly increasing ints, "
                             f"first = 0, all <= {n_bars})")
        if it.get("alt_bar") is not None and not (0 < it["alt_bar"] <= n_bars):
            raise SystemExit(f"{path}: alt_bar {it['alt_bar']} out of range for '{it['label']}'")
        prev = b
    if len({it["label"] for it in items}) != len(items):
        raise SystemExit(f"{path}: labels must be unique")
    start = lambda b: 0.0 if b == 0 else down_t[b]
    secs = []
    for k, it in enumerate(items):
        t0 = start(it["bar"])
        t1 = start(items[k + 1]["bar"]) if k + 1 < len(items) else duration
        sec = {"idx": k, "label": it["label"], "t_start": t0, "t_end": t1, "t_start_raw": t0,
               "t_end_raw": t1, "snap_beats": 0.0, "basis": it.get("basis")}
        if it.get("alt_bar") is not None:
            sec["alt_bar"], sec["alt_t"], sec["alt_basis"] = it["alt_bar"], r3(start(it["alt_bar"])), it.get("alt_basis")
        secs.append(sec)
    return secs


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allin1", required=True)
    ap.add_argument("--energy", required=True, help="librosa.json")
    ap.add_argument("--srt", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--sections", help="手動段落覆寫 JSON（每段從 bar 的首拍開始）；未給則用 allin1 段落")
    ap.add_argument("--beat-corrections", help="beat_corrections.json：替換 allin1 拍點（bar 38 / 49 等）")
    ap.add_argument("--bpm-mode", choices=["auto", "as-is", "double", "halve"], default="auto")
    args = ap.parse_args()

    warnings = []
    a, lib = load_json(args.allin1), load_json(args.energy)
    cues = parse_srt(Path(args.srt).read_text(encoding="utf-8"))
    duration = float(lib["duration_sec"])
    n_frames = frame(duration)

    # ---- sha256
    sources = {}
    for key, p in (("audio", args.audio), ("allin1", args.allin1), ("energy", args.energy), ("srt", args.srt)):
        if Path(p).exists():
            sources[key] = {"path": str(p), "sha256": sha256(p)}
        else:
            sources[key] = {"path": str(p), "sha256": None}
            warnings.append(f"{key} file not found, sha256 not recorded: {p}")
    if sources["audio"]["sha256"] and lib.get("source_sha256") and lib["source_sha256"] != sources["audio"]["sha256"]:
        warnings.append("librosa.json source_sha256 does not match the audio file")

    # ---- energy
    rms, onset = list(lib["rms_norm"]), list(lib["onset_norm"])
    for name, arr in (("rms_norm", rms), ("onset_norm", onset)):
        if len(arr) != n_frames:
            warnings.append(f"{name} length {len(arr)} != n_frames {n_frames}; padded/trimmed")
    rms = (rms + [0.0] * n_frames)[:n_frames]
    onset = (onset + [0.0] * n_frames)[:n_frames]

    # ---- beats / bars
    times, pos = beats_from_allin1(a, warnings)
    times, pos = transform_grid(times, pos, args.bpm_mode)
    corrected = [False] * len(times)
    corr_meta = None
    if args.beat_corrections:
        times, pos, corrected, regions = apply_beat_corrections(args.beat_corrections, times, pos)
        corr_meta = {"path": str(args.beat_corrections), "sha256": sha256(args.beat_corrections), "regions": regions}
        sources["beat_corrections"] = {"path": corr_meta["path"], "sha256": corr_meta["sha256"]}
    bpm, bpm_source, alts = decide_bpm(float(a["bpm"]), lib, args.bpm_mode, warnings)
    beat_sec = 60.0 / bpm
    bar_of = assign_bars(pos)
    if not any(p == 1 for p in pos):
        raise SystemExit("no downbeat found in beat grid")
    beats = [{"i": i, "t": r3(t), "frame": frame(t), "bar": b, "beat_in_bar": p, "is_downbeat": p == 1,
              **({"corrected": True} if c else {})}
             for i, (t, p, b, c) in enumerate(zip(times, pos, bar_of, corrected))]
    n_bars = max(bar_of)
    down_t = [None] * (n_bars + 1)
    count = [0] * (n_bars + 1)
    for t, b, p in zip(times, bar_of, pos):
        count[b] += 1
        if p == 1:
            down_t[b] = t
    if count[0]:
        warnings.append(f"{count[0]} pickup beat(s) before the first downbeat (bar 0)")
    irregular = [b for b in range(1, n_bars + 1) if count[b] != BEATS_PER_BAR]
    # 修正區域涉及的小節：該區域任一修正拍所在小節，或其緊接的前一小節
    region_of = {}
    if args.beat_corrections:
        for c in load_json(args.beat_corrections)["corrections"]:
            for t in c["new_beats_t"]:
                k = min(range(len(times)), key=lambda j: abs(times[j] - t))
                region_of.setdefault(bar_of[k], c["region"])
    irregular_bars, unexplained = [], []
    for b in irregular:
        if b in region_of:
            reason = "beat_corrections:" + region_of[b]
        elif b == n_bars:
            reason = "song end"
        else:
            reason = "unexplained"
            unexplained.append(b)
        irregular_bars.append({"bar": b, "beats": count[b], "reason": reason})
    if unexplained:
        warnings.append("irregular bars (!= 4 beats) with no known reason: " +
                        ", ".join(f"bar {b} has {count[b]}" for b in unexplained[:20]) +
                        (" ..." if len(unexplained) > 20 else ""))
    last_end = min(times[-1] + beat_sec, duration)
    bar_t = [down_t[b] for b in range(1, n_bars + 1)] + [last_end]
    bar_starts = bar_t[:-1]

    def bar_at(t):
        """t 所在小節；首個首拍之前為 0。"""
        return bisect.bisect_right(bar_starts, t + 1e-9)

    # ---- sections
    secs_a = build_sections(a["segments"], [t for t, p in zip(times, pos) if p == 1], duration, beat_sec, warnings)
    if args.sections:
        secs = override_sections(args.sections, down_t, duration)
        sections_source = "override:" + str(args.sections)
        sources["sections"] = {"path": str(args.sections), "sha256": sha256(args.sections)}
    else:
        secs, sections_source = secs_a, "allin1"
    sec_starts = [s["t_start"] for s in secs]

    def section_at(t):
        return secs[max(0, bisect.bisect_right(sec_starts, t + 1e-9) - 1)]

    # ---- lyrics
    lyrics = []
    for c in cues:
        mid = (c["t_start"] + c["t_end"]) / 2
        lyrics.append({"id": c["id"], "t_start": r3(c["t_start"]), "t_end": r3(c["t_end"]),
                       "frame_start": frame(c["t_start"]), "frame_end": frame(c["t_end"]),
                       "text": c["text"], "bar_start": bar_at(c["t_start"]),
                       "bar_end": bar_at(c["t_end"] - 1e-3), "section": section_at(mid)["label"]})

    # ---- bars
    bars, raw_e, raw_o = [], [], []
    for k in range(1, n_bars + 1):
        fs = frame(bar_t[k - 1])
        fe = n_frames if k == n_bars else frame(bar_t[k])
        seg_r, seg_o = rms[fs:fe], onset[fs:fe]
        raw_e.append(mean(seg_r))
        raw_o.append(mean(seg_o))
        bars.append({"bar": k, "t_start": r3(bar_t[k - 1]), "t_end": r3(bar_t[k]),
                     "frame_start": fs, "frame_end": fe,
                     "section": section_at(bar_t[k - 1])["label"],
                     "energy": 0.0, "energy_peak": round(10 * max(seg_r), 1) if seg_r else 0.0,
                     "onset": 0.0,
                     "lyric_ids": [l["id"] for l in lyrics if l["bar_start"] <= k <= l["bar_end"]]})
    # 小節能量以全曲最響小節為 10 重新縮放（bar mean / max bar mean x 10）
    max_e, max_o = max(raw_e, default=0) or 1.0, max(raw_o, default=0) or 1.0
    norm_e = [x / max_e * 10 for x in raw_e]
    for b, ne, no in zip(bars, norm_e, raw_o):
        b["energy"] = round(ne, 1)
        b["onset"] = round(no / max_o * 10, 1)
    if frame(bar_t[-1]) < n_frames - 1:
        warnings.append(f"last bar ends at {bar_t[-1]:.3f}s but song is {duration:.3f}s; "
                        f"last bar frame_end extended to n_frames")

    # ---- sections output
    def render(secs):
        out_secs = []
        for s in secs:
            fs, fe = frame(s["t_start"]), (n_frames if s["idx"] == len(secs) - 1 else frame(s["t_end"]))
            b0, b1 = bar_at(s["t_start"]), bar_at(s["t_end"] - 1e-3)
            o = {"idx": s["idx"], "label": s["label"], "t_start": r3(s["t_start"]),
                 "t_end": r3(s["t_end"]), "t_start_raw": r3(s["t_start_raw"]),
                 "t_end_raw": r3(s["t_end_raw"]), "snap_beats": s["snap_beats"],
                 "frame_start": fs, "frame_end": fe, "bar_start": b0, "bar_end": b1,
                 "n_bars": max(0, b1 - max(b0, 1) + 1),
                 "energy": round(mean(norm_e[max(b0, 1) - 1:b1]) if b1 >= max(b0, 1)
                                 else mean(rms[fs:fe]) / max_e * 10, 1)}
            for k in ("basis", "alt_bar", "alt_t", "alt_basis"):
                if k in s:
                    o[k] = s[k]
            out_secs.append(o)
        return out_secs
    sections, sections_allin1 = render(secs), render(secs_a)

    # ---- hits（accent points，取自 --sections 的頂層 hits）
    hits = []
    if args.sections:
        for h in load_json(args.sections).get("hits", []):
            hb = h.get("bar")
            if not isinstance(hb, int) or not (1 <= hb <= n_bars):
                raise SystemExit(f"{args.sections}: hit bar {hb!r} out of range 1..{n_bars}")
            bar = bars[hb - 1]
            bar["hit"] = h["type"]
            hits.append({"bar": hb, "type": h["type"], "basis": h.get("basis"), "t": bar["t_start"],
                         "frame": bar["frame_start"], "section": bar["section"]})

    meta = {"fps": FPS, "duration_sec": r3(duration), "n_frames": n_frames,
            "bpm": round(bpm, 3), "bpm_source": bpm_source, "bpm_alternatives": alts,
            "beats_per_bar": BEATS_PER_BAR, "beat_sec": r3(beat_sec), "bar_sec": r3(beat_sec * BEATS_PER_BAR),
            "beat_frames": round(beat_sec * FPS, 3), "bar_frames": round(beat_sec * FPS * BEATS_PER_BAR, 3),
            "frame_rule": "frame = round(t * 24)",
            "energy_scale": "bar/section energy & onset = bar mean rms_norm (onset_norm) / max bar mean x 10; "
                            "energy_peak = max rms_norm x 10 within bar (raw scale)", "sources": sources,
            "sections_source": sections_source, "irregular_bars": irregular_bars,
            **({"beat_corrections": corr_meta} if corr_meta else {}), "warnings": warnings, "generated_by": "tools/build_timeline.py"}
    out = {"meta": meta, "beats": beats, "bars": bars, "sections": sections, "sections_allin1": sections_allin1, "hits": hits, "lyrics": lyrics,
           "energy": {"fps": FPS, "rms_norm": [round(x, 4) for x in rms],
                      "onset_norm": [round(x, 4) for x in onset]}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.output}: {len(beats)} beats, {n_bars} bars, {len(sections)} sections, "
          f"{len(lyrics)} lyrics, bpm {bpm:g} ({bpm_source}), {n_frames} frames")
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    sys.exit(main())
