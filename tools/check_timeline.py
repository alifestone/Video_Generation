#!/usr/bin/env python3
"""驗證 analysis/timeline.json（只用標準函式庫）；有 FAIL 時結束碼為 1。

python3 tools/check_timeline.py analysis/timeline.json [--srt audio/lyrics.srt] [--fixture]
--fixture：測試用資料，真實歌曲才有的期望值（3223 幀、32 行歌詞）只警告不失敗。
"""
import argparse, json, statistics, sys
from pathlib import Path

from build_timeline import parse_srt

REAL_FRAMES, REAL_CUES = 3223, 32


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("timeline")
    ap.add_argument("--srt", help="預設用 timeline 記錄的 SRT 路徑")
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()

    tl = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    m, beats, bars, secs, lyr, en = (tl[k] for k in ("meta", "beats", "bars", "sections", "lyrics", "energy"))
    fails, warns, results = [], [], []

    def check(name, ok, detail="", soft=False):
        results.append(("PASS" if ok else ("WARN" if soft else "FAIL"), name, detail))
        if not ok:
            (warns if soft else fails).append(name)

    n, fps = m["n_frames"], m["fps"]
    check("n_frames == round(duration*24)", n == round(m["duration_sec"] * fps), f"{n}")
    check(f"n_frames == {REAL_FRAMES} (real song)", n == REAL_FRAMES, f"{n}", soft=args.fixture)

    ts = [b["t"] for b in beats]
    check("beats strictly increasing", all(a < b for a, b in zip(ts, ts[1:])))
    check("beat.frame == round(t*24)", all(b["frame"] == round(b["t"] * fps) for b in beats))
    n_in_bar = {}
    for b in beats:
        n_in_bar[b["bar"]] = n_in_bar.get(b["bar"], 0) + 1
    check("beat_in_bar in 1..(beats in bar) and is_downbeat consistent",
          all(1 <= b["beat_in_bar"] <= max(n_in_bar[b["bar"]], 4 if b["bar"] == 0 else 0)
              and b["is_downbeat"] == (b["beat_in_bar"] == 1) for b in beats))
    irr = {x["bar"]: x for x in m.get("irregular_bars", [])}
    odd = [k for k, c in n_in_bar.items() if k >= 1 and c != 4]
    check("every non-4-beat bar is listed in meta.irregular_bars", all(k in irr and irr[k]["beats"] == n_in_bar[k] for k in odd),
          ", ".join(f"bar {k}={n_in_bar[k]}({irr[k]['reason'] if k in irr else 'unlisted'})" for k in sorted(odd)))
    check("irregular_bars all explained", all(x["reason"] != "unexplained" for x in irr.values()), soft=True)
    check("beat.bar non-decreasing", all(a["bar"] <= b["bar"] for a, b in zip(beats, beats[1:])))
    check("BPM within 60-240", 60 <= m["bpm"] <= 240, f"{m['bpm']}")
    if len(ts) > 1:
        med = statistics.median(b - a for a, b in zip(ts, ts[1:]))
        check("median beat interval within 2% of 60/bpm", abs(med - 60 / m["bpm"]) <= 0.02 * 60 / m["bpm"],
              f"median {med:.4f}s vs {60 / m['bpm']:.4f}s")

    downs = [b for b in beats if b["is_downbeat"]]
    check("bars numbered 1..N", [b["bar"] for b in bars] == list(range(1, len(bars) + 1)), f"{len(bars)} bars")
    check("bars match downbeats", len(downs) == len(bars) and all(
        d["frame"] == b["frame_start"] for d, b in zip(downs, bars)))
    check("bars contiguous", all(a["frame_end"] == b["frame_start"] for a, b in zip(bars, bars[1:])))
    check("bar frame ranges positive", all(b["frame_end"] > b["frame_start"] for b in bars))
    check("bars tile [first downbeat, n_frames)",
          bool(bars) and bars[0]["frame_start"] == (downs[0]["frame"] if downs else -1) and bars[-1]["frame_end"] == n)
    check("bar energy in 0..10", all(0 <= b[k] <= 10 for b in bars for k in ("energy", "energy_peak", "onset")))

    def tiles(ss):
        return (bool(ss) and ss[0]["frame_start"] == 0 and ss[-1]["frame_end"] == n and
                all(a["frame_end"] == b["frame_start"] for a, b in zip(ss, ss[1:])) and
                all(s["frame_end"] > s["frame_start"] for s in ss))
    check("sections tile [0, n_frames)", tiles(secs), m.get("sections_source", ""))
    alt = tl.get("sections_allin1")
    if alt is not None:
        check("sections_allin1 tile [0, n_frames)", tiles(alt))
    check("section labels unique", len({s["label"] for s in secs}) == len(secs))
    check("section snap within 2 beats", all(abs(s["snap_beats"]) <= 2 for s in secs),
          "max %.2f" % max(abs(s["snap_beats"]) for s in secs), soft=True)

    hits = tl.get("hits", [])
    bar_by_n = {b["bar"]: b for b in bars}
    down_frames = {d["frame"] for d in downs}
    check("hits on bar downbeats inside their section", all(
        h["bar"] in bar_by_n and h["frame"] == bar_by_n[h["bar"]]["frame_start"] and h["frame"] in down_frames
        and bar_by_n[h["bar"]]["section"] == h["section"] and bar_by_n[h["bar"]].get("hit") == h["type"]
        and any(s["label"] == h["section"] and s["frame_start"] <= h["frame"] < s["frame_end"] for s in secs)
        for h in hits), f"{len(hits)} hit(s)")

    srt_path = args.srt or m["sources"]["srt"]["path"]
    if Path(srt_path).exists():
        cues = parse_srt(Path(srt_path).read_text(encoding="utf-8"))
        by_id = {l["id"]: l for l in lyr}
        check("every SRT cue present in lyrics", len(cues) == len(lyr) and all(
            c["id"] in by_id and by_id[c["id"]]["text"] == c["text"] for c in cues), f"{len(cues)} cues")
        check(f"SRT has {REAL_CUES} cues (real file)", len(cues) == REAL_CUES, f"{len(cues)}", soft=args.fixture)
    else:
        check("SRT file readable", False, srt_path, soft=True)
    nb = len(bars)
    check("lyric bar refs valid", all(0 <= l["bar_start"] <= l["bar_end"] <= nb for l in lyr))
    check("lyric frames valid", all(0 <= l["frame_start"] < l["frame_end"] <= n for l in lyr))
    labels = {s["label"] for s in secs}
    check("lyric sections valid", all(l["section"] in labels for l in lyr))
    check("bars' lyric_ids match lyrics", all(
        (l["id"] in b["lyric_ids"]) == (l["bar_start"] <= b["bar"] <= l["bar_end"]) for l in lyr for b in bars))
    check("energy arrays length == n_frames", len(en["rms_norm"]) == n and len(en["onset_norm"]) == n,
          f"{len(en['rms_norm'])}/{len(en['onset_norm'])}")

    print(f"timeline: {args.timeline}  bpm {m['bpm']:g} ({m['bpm_source']})  {n} frames  {len(bars)} bars")
    for status, name, detail in results:
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    def table(title, ss):
        print(f"\n{title}")
        print(f"{'idx':>3} {'label':<12} {'start-end':<15} {'frames':<13} {'bars':>9} {'n':>3} {'energy':>6} {'snap':>6}")
        for s in ss:
            print(f"{s['idx']:>3} {s['label']:<12} {mmss(s['t_start'])}-{mmss(s['t_end']):<9} "
                  f"{s['frame_start']}-{s['frame_end']:<7} {s['bar_start']:>4}-{s['bar_end']:<4} {s['n_bars']:>3} "
                  f"{s['energy']:>6.1f} {s['snap_beats']:>+6.2f}" +
                  (f"  alt bar {s['alt_bar']}" if "alt_bar" in s else ""))
    table(f"sections ({m.get('sections_source', 'allin1')})", secs)
    if alt is not None:
        table("sections_allin1 (reference)", alt)
    if hits:
        print("\nhits")
        for h in hits:
            print(f"  bar {h['bar']:<3} {h['type']:<8} {mmss(h['t'])} frame {h['frame']:<5} in {h['section']}")
    if m.get("irregular_bars"):
        print("\nirregular bars")
        for x in m["irregular_bars"]:
            print(f"  bar {x['bar']:<3} {x['beats']} beats  {x['reason']}")
    if m.get("warnings"):
        print("\nbuild warnings:")
        for w in m["warnings"]:
            print("  -", w)
    print(f"\n{'FAIL' if fails else 'PASS'}: {len(fails)} failed, {len(warns)} warnings")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
