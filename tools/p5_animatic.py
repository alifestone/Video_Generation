#!/usr/bin/env python3
"""P5 animatic：shotlist.json ＋ 每鏡草圖 ＋ audio/song.wav → edit/animatic.mp4 與 edit/animatic.kdenlive。

~/creative/ComfyUI/comfyui/bin/python tools/p5_animatic.py            # 產生 mp4 ＋ kdenlive ＋ 驗證
~/creative/ComfyUI/comfyui/bin/python tools/p5_animatic.py --verify-only   # 只驗證既有 animatic.mp4
選項：--also-abs（另存 animatic_abs.kdenlive）、--drafts DIR（預設 edit/drafts）、--no-card（不疊資訊卡）、--no-mp4／--no-kdenlive、--relative（kdenlive 用相對路徑）、
      --shotlist／--out-dir／--audio（測試用）。需要 PIL（用 ComfyUI 的 python）、ffmpeg、ffprobe。

mp4：1280×720、24fps、幀數＝shotlist.meta.n_frames；每鏡的圖放在絕對起訖幀（純硬切，不做轉場；transition_in 只印在資訊卡）。
  實作：Python 逐幀餵 rawvideo 給 ffmpeg（每鏡寫 n_frames 次），所以幀數與切點幀號由程式保證，不靠 concat 的時間換算。
  沒有草圖的鏡頭用灰底（中央淡淡的鏡號）。資訊卡（左上）：鏡號、段落、景別／角度、運鏡、model、tier、transition_in、幀區間。
kdenlive：MLT XML（profile atsc_720p_24），音軌 A1 放原曲、影像軌 V1 依絕對幀排每鏡一個 clip（有草圖＝圖片 clip，沒有＝灰色 color clip）；
  clip 名稱＝「鏡號 段落 景別」。本機沒有 Kdenlive／melt，只做 xmllint 與「把 XML 讀回來核對每個 clip 的起訖幀」的驗證。
驗證：ffprobe 幀數與時長；整支解碼成小圖逐幀比差，畫面「換圖」的幀號集合必須等於 shotlist 的 start_frame 集合；抽 5 個切點印出前後幀差。
"""
import argparse, json, os, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 1280, 720, 24
FONT_FILE = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"
GRAY = (58, 58, 62)


def load_font(path, size):
    """Noto Sans CJK TC（ttc 內有 JP/KR/SC/TC/HK 五個 face，依名稱找 TC）。"""
    for i in range(6):
        try:
            f = ImageFont.truetype(path, size, index=i)
        except Exception:
            break
        if "TC" in " ".join(f.getname()):
            return f
    return ImageFont.truetype(path, size)


def fit(img):
    img = img.convert("RGB")
    s = max(W / img.width, H / img.height)
    img = img.resize((max(W, round(img.width * s)), max(H, round(img.height * s))), Image.LANCZOS)
    x, y = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((x, y, x + W, y + H))


def card_lines(s):
    cam = s["camera"]
    n = s["n_frames"]
    t = s["transition_in"]
    tr = t["type"] + (f" {t['frames']}f" if t.get("frames") else "")
    return [
        f"{s['id']}  {s['section']}",
        f"{s['shot_size']} / {s['angle']}   cam: {cam['move']} {cam['speed']} ({cam['realize']})",
        f"model: {s['model'] or '-'}   tier: {s['tier'] or '-'}",
        f"in: {tr}   f{s['start_frame']}–{s['end_frame']} ({n}f, {n / FPS:.2f}s)",
    ]


def render_shot(s, draft_path, card=True):
    if draft_path:
        img = fit(Image.open(draft_path))
    else:
        img = Image.new("RGB", (W, H), GRAY)
        d = ImageDraw.Draw(img)
        big = load_font(FONT_BOLD, 200)
        d.text((W // 2, H // 2), s["id"], font=big, fill=(84, 84, 90), anchor="mm")
    if card:
        lines = card_lines(s)
        f0, f1 = load_font(FONT_BOLD, 30), load_font(FONT_FILE, 20)
        pad, lh0, lh1 = 10, 38, 27
        wmax = max([f0.getlength(lines[0])] + [f1.getlength(x) for x in lines[1:]])
        box = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(box)
        bd.rounded_rectangle((12, 12, 12 + wmax + 2 * pad, 12 + lh0 + lh1 * (len(lines) - 1) + 2 * pad), 8, fill=(0, 0, 0, 170))
        img = Image.alpha_composite(img.convert("RGBA"), box).convert("RGB")
        d = ImageDraw.Draw(img)
        y = 12 + pad
        d.text((12 + pad, y), lines[0], font=f0, fill=(255, 255, 255))
        y += lh0
        for x in lines[1:]:
            d.text((12 + pad, y), x, font=f1, fill=(225, 235, 245))
            y += lh1
    return img


def draft_for(drafts, sid):
    p = drafts / f"{sid}.png"
    return p if p.exists() else None


def make_mp4(shots, drafts, audio, out, n_frames, card):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-g", "24", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    done = 0
    for s in shots:
        raw = render_shot(s, draft_for(drafts, s["id"]), card).tobytes()
        for _ in range(s["n_frames"]):
            p.stdin.write(raw)
        done += s["n_frames"]
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg 編碼失敗")
    assert done == n_frames, (done, n_frames)


# ---------- Kdenlive ----------
def prop(parent, name, value=""):
    e = ET.SubElement(parent, "property", name=name)
    e.text = str(value)
    return e


def make_kdenlive(shots, drafts, audio, out, n_frames, relative):
    base = out.parent

    def res(p):
        return str(Path(os.path.relpath(p, base))) if relative else str(Path(p).resolve())

    # 相對路徑版不寫 root（MLT 載入 XML 檔時以檔案所在資料夾為 root，專案搬位置仍可開）；絕對路徑版寫 root
    attrs = dict(LC_NUMERIC="C", producer="main_bin", version="7.0.0")
    if not relative:
        attrs["root"] = str(base.resolve())
    mlt = ET.Element("mlt", **attrs)
    ET.SubElement(mlt, "profile", colorspace="709", description="HD 720p 24 fps", display_aspect_den="9", display_aspect_num="16",
                  frame_rate_den="1", frame_rate_num=str(FPS), height=str(H), progressive="1", sample_aspect_den="1", sample_aspect_num="1",
                  width=str(W))
    last = n_frames - 1
    # 黑底與各 clip（bin）
    blk = ET.SubElement(mlt, "producer", id="black_track", **{"in": "0", "out": "2147483646"})
    for k, v in [("length", 2147483647), ("eof", "continue"), ("resource", "black"), ("aspect_ratio", 1), ("mlt_service", "color"),
                 ("kdenlive:playlistid", "black_track"), ("mlt_image_format", "rgba"), ("set.test_audio", 0)]:
        prop(blk, k, v)
    next_id = 2
    audio_id = next_id
    next_id += 1
    pa = ET.SubElement(mlt, "producer", id="producer_audio", **{"in": "0", "out": str(last)})
    for k, v in [("length", n_frames + 1), ("eof", "pause"), ("resource", res(audio)), ("audio_index", 0), ("video_index", -1),
                 ("mute_on_pause", 0), ("mlt_service", "avformat-novalidate"), ("seekable", 1), ("aspect_ratio", 1),
                 ("kdenlive:clipname", audio.name), ("kdenlive:clip_type", 1), ("kdenlive:id", audio_id), ("kdenlive:folderid", -1),
                 ("xml", "was here")]:
        prop(pa, k, v)
    clips = []   # (producer_id, bin_id, shot)
    for s in shots:
        d = draft_for(drafts, s["id"])
        pid, bid = f"producer_{s['id']}", next_id
        next_id += 1
        n = s["n_frames"]
        p = ET.SubElement(mlt, "producer", id=pid, **{"in": "0", "out": str(n - 1)})
        name = f"{s['id']} {s['section']} {s['shot_size']}"
        if d:
            items = [("length", n), ("eof", "pause"), ("resource", res(d)), ("ttl", 1), ("aspect_ratio", 1), ("meta.media.progressive", 1),
                     ("seekable", 1), ("mlt_service", "qimage"), ("kdenlive:duration", f"{n}"), ("kdenlive:clip_type", 2)]
        else:
            items = [("length", n), ("eof", "pause"), ("resource", "0x3a3a3eff"), ("aspect_ratio", 1), ("mlt_service", "color"),
                     ("set.test_audio", 0), ("kdenlive:duration", f"{n}"), ("kdenlive:clip_type", 5)]
        for k, v in items + [("kdenlive:clipname", name), ("kdenlive:id", bid), ("kdenlive:folderid", -1), ("xml", "was here")]:
            prop(p, k, v)
        clips.append((pid, bid, s))

    def track_pair(idx, audio_track):
        main = ET.SubElement(mlt, "playlist", id=f"playlist{2 * idx}")
        if audio_track:
            prop(main, "kdenlive:audio_track", 1)
        sec = ET.SubElement(mlt, "playlist", id=f"playlist{2 * idx + 1}")
        if audio_track:
            prop(sec, "kdenlive:audio_track", 1)
        return main

    # 音軌 A1
    pl = track_pair(0, True)
    e = ET.SubElement(pl, "entry", **{"in": "0", "out": str(last), "producer": "producer_audio"})
    prop(e, "kdenlive:id", audio_id)
    t0 = ET.SubElement(mlt, "tractor", id="tractor0", **{"in": "0", "out": str(last)})
    for k, v in [("kdenlive:audio_track", 1), ("kdenlive:trackheight", 67), ("kdenlive:timeline_active", 1), ("kdenlive:collapsed", 0),
                 ("kdenlive:thumbs_format", ""), ("kdenlive:audio_rec", "")]:
        prop(t0, k, v)
    ET.SubElement(t0, "track", hide="video", producer="playlist0")
    ET.SubElement(t0, "track", hide="video", producer="playlist1")
    for fid, svc, extra in [("filter0", "volume", [("window", 75), ("max_gain", "20dB"), ("internal_added", 237), ("disable", 1)]),
                            ("filter1", "panner", [("channel", -1), ("internal_added", 237), ("start", 0.5), ("disable", 1)])]:
        f = ET.SubElement(t0, "filter", id=fid)
        prop(f, "mlt_service", svc)
        for k, v in extra:
            prop(f, k, v)
    # 影像軌 V1：依絕對幀排，檔案裡每個 clip 的位置＝前面各 clip 長度總和（鏡頭無縫相接，不放 blank）
    pl = track_pair(1, False)
    pos = 0
    for pid, bid, s in clips:
        assert pos == s["start_frame"], (s["id"], pos, s["start_frame"])
        e = ET.SubElement(pl, "entry", **{"in": "0", "out": str(s["n_frames"] - 1), "producer": pid})
        prop(e, "kdenlive:id", bid)
        pos += s["n_frames"]
    t1 = ET.SubElement(mlt, "tractor", id="tractor1", **{"in": "0", "out": str(last)})
    for k, v in [("kdenlive:trackheight", 67), ("kdenlive:timeline_active", 1), ("kdenlive:thumbs_format", ""), ("kdenlive:audio_rec", "")]:
        prop(t1, k, v)
    ET.SubElement(t1, "track", hide="audio", producer="playlist2")
    ET.SubElement(t1, "track", hide="audio", producer="playlist3")
    # bin
    mb = ET.SubElement(mlt, "playlist", id="main_bin")
    for k, v in [("kdenlive:docproperties.audiotargettrack", 1), ("kdenlive:docproperties.decimalPoint", "."),
                 ("kdenlive:docproperties.documentid", str(int(time.time() * 1000))), ("kdenlive:docproperties.enableTimelineZone", 0),
                 ("kdenlive:docproperties.enableexternalproxy", 0), ("kdenlive:docproperties.enableproxy", 0),
                 ("kdenlive:docproperties.generateimageproxy", 0), ("kdenlive:docproperties.generateproxy", 0),
                 ("kdenlive:docproperties.kdenliveversion", "22.12.3"), ("kdenlive:docproperties.profile", "atsc_720p_24"),
                 ("kdenlive:docproperties.projectfolder", str(base.resolve())), ("kdenlive:docproperties.seekOffset", 0),
                 ("kdenlive:docproperties.version", "1.0"), ("kdenlive:docproperties.videotargettrack", 2),
                 ("kdenlive:docproperties.timelinezone", f"0;{last}"),
                 ("kdenlive:expandedFolders", ""), ("kdenlive:binZoom", 4), ("xml_retain", 1)]:
        prop(mb, k, v)
    ET.SubElement(mb, "entry", **{"in": "0", "out": str(last), "producer": "producer_audio"})
    for pid, bid, s in clips:
        ET.SubElement(mb, "entry", **{"in": "0", "out": str(s["n_frames"] - 1), "producer": pid})
    # 時間軸 tractor（最後一個 tractor）
    tm = ET.SubElement(mlt, "tractor", id="tractor2", **{"in": "0", "out": str(last)})
    prop(tm, "kdenlive:projectTrackCount", 2)
    ET.SubElement(tm, "track", producer="black_track")
    ET.SubElement(tm, "track", producer="tractor0")
    ET.SubElement(tm, "track", producer="tractor1")
    for tid, b, svc, extra in [("transition0", 1, "mix", [("kdenlive_id", "mix"), ("accepts_blanks", 1), ("sum", 1)]),
                               ("transition1", 2, "qtblend", [("kdenlive_id", "qtblend"), ("compositing", 0), ("distort", 0), ("rotate_center", 0)])]:
        tr = ET.SubElement(tm, "transition", id=tid)
        for k, v in [("a_track", 0), ("b_track", b), ("mlt_service", svc), ("internal_added", 237), ("always_active", 1)] + extra:
            prop(tr, k, v)
    ET.indent(mlt, space=" ")
    ET.ElementTree(mlt).write(out, encoding="utf-8", xml_declaration=True)
    return len(clips)


def verify_kdenlive(path, shots, n_frames):
    """XML 讀回來核對：V1 的 clip 依序累加的起訖幀＝shotlist；音軌長度＝n_frames。"""
    r = subprocess.run(["xmllint", "--noout", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("xmllint 失敗：" + r.stderr[:500])
    root = ET.parse(path).getroot()
    pl = {p.get("id"): p for p in root.findall("playlist")}
    prods = {p.get("id"): p for p in root.findall("producer")}
    assert int(root.find("profile").get("frame_rate_num")) == FPS and root.find("profile").get("width") == str(W)
    v1 = pl["playlist2"]
    pos, bad = 0, 0
    entries = v1.findall("entry")
    assert len(entries) == len(shots), (len(entries), len(shots))
    for e, s in zip(entries, shots):
        n = int(e.get("out")) - int(e.get("in")) + 1
        if pos != s["start_frame"] or pos + n != s["end_frame"] or e.get("producer") not in prods:
            bad += 1
        pos += n
    a = pl["playlist0"].find("entry")
    ok_audio = int(a.get("out")) - int(a.get("in")) + 1 == n_frames
    tracks = [t.get("producer") for t in root.findall("tractor")[-1].findall("track")]
    print(f"kdenlive：xmllint 通過；V1 {len(entries)} 個 clip 累加後終點 {pos}（應 {n_frames}），起訖不符 {bad} 個；A1 長度{'正確' if ok_audio else '錯誤'}；時間軸 tracks={tracks}")
    if bad or pos != n_frames or not ok_audio:
        sys.exit("kdenlive 核對失敗")


# ---------- 驗證 mp4 ----------
def ffprobe(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames,width,height,r_frame_rate,codec_name:format=duration", "-of", "json", str(mp4)],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    return j["streams"][0], float(j["format"]["duration"])


def decode_small(mp4, w=320, h=180):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-map", "0:v:0", "-vf", f"scale={w}:{h}:flags=area,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w).astype(np.int16)


def verify_mp4(mp4, shots, n_frames, duration_sec):
    st, dur = ffprobe(mp4)
    nb = int(st["nb_read_frames"])
    print(f"ffprobe：{st['codec_name']} {st['width']}x{st['height']} {st['r_frame_rate']}，幀數 {nb}（應 {n_frames}），時長 {dur:.3f}s（timeline {duration_sec}s）")
    fails = []
    if nb != n_frames or (st["width"], st["height"]) != (W, H) or st["r_frame_rate"] != f"{FPS}/1":
        fails.append("影片規格或幀數不符")
    if abs(dur - duration_sec) > 0.05:
        fails.append("時長誤差 > 0.05s")
    fr = decode_small(mp4)
    diff = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))        # diff[i] = 幀 i+1 與幀 i 的差
    # 每鏡取中間幀當該鏡的參考圖；每一幀都必須離自己鏡的參考圖比離前後鏡的參考圖更近（對 H.264 雜訊不敏感）。
    # 切點幀號因此被逐幀驗證：start_frame 之前一幀屬於前一鏡、start_frame 起屬於本鏡。
    mids = [fr[(s["start_frame"] + s["end_frame"]) // 2] for s in shots]
    dist = lambda x, ref: np.abs(x - ref).mean(axis=(1, 2))
    wrong, ambiguous = [], 0
    for k, s in enumerate(shots):
        x = fr[s["start_frame"]:s["end_frame"]]
        own = dist(x, mids[k])
        for j in (k - 1, k + 1):
            if 0 <= j < len(shots):
                if float(np.abs(mids[k] - mids[j]).mean()) < 0.05:   # 相鄰兩鏡畫面一模一樣，無法用畫面判斷
                    ambiguous += 1
                    continue
                bad = np.nonzero(own >= dist(x, mids[j]))[0]
                wrong += [(s["id"], int(s["start_frame"] + i)) for i in bad]
    print(f"逐幀歸屬檢查：{len(shots)} 鏡、{n_frames} 幀，歸屬錯誤 {len(wrong)} 幀{'（' + str(wrong[:5]) + '）' if wrong else ''}，相鄰畫面相同而略過 {ambiguous} 對")
    if wrong:
        fails.append("有幀的畫面不屬於 shotlist 指定的鏡頭（切點幀號不符）")
    expect = [s["start_frame"] for s in shots[1:]]
    idx = np.linspace(0, len(expect) - 1, 5).round().astype(int)
    for i in idx:
        f = expect[i]
        s = shots[i + 1]
        within = float(diff[f]) if f < len(diff) else 0.0   # 切點後下一幀（同鏡內，應≈0）
        print(f"  抽查 {s['id']} start_frame={f}：幀 {f - 1}→{f} 差 {diff[f - 1]:.2f}（換圖），幀 {f}→{f + 1} 差 {within:.2f}（同鏡內）")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shotlist", default="shotlist.json")
    ap.add_argument("--drafts", default="edit/drafts")
    ap.add_argument("--out-dir", default="edit")
    ap.add_argument("--audio", default="audio/song.wav")
    ap.add_argument("--no-card", action="store_true")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--no-kdenlive", action="store_true")
    ap.add_argument("--relative", action="store_true", help="kdenlive 內的素材用相對於專案檔的路徑")
    ap.add_argument("--also-abs", action="store_true", help="另存絕對路徑版 animatic_abs.kdenlive（搭配 --relative：animatic.kdenlive 相對、_abs 絕對）")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    R = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    sl = json.loads(R(a.shotlist).read_text())
    shots, n_frames = sl["shots"], sl["meta"]["n_frames"]
    out = R(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp4, kd = out / "animatic.mp4", out / "animatic.kdenlive"
    drafts = R(a.drafts)
    have = sum(1 for s in shots if draft_for(drafts, s["id"]))
    print(f"{len(shots)} 鏡，{n_frames} 幀；有草圖 {have} 鏡，灰底 {len(shots) - have} 鏡")
    fails = []
    if not a.verify_only and not a.no_mp4:
        t0 = time.time()
        make_mp4(shots, drafts, R(a.audio), mp4, n_frames, not a.no_card)
        print(f"寫入 {mp4}（{mp4.stat().st_size / 1e6:.1f} MB，{time.time() - t0:.0f}s）")
    if not a.verify_only and not a.no_kdenlive:
        n = make_kdenlive(shots, drafts, R(a.audio), kd, n_frames, a.relative)
        print(f"寫入 {kd}（{n} 個 clip，{'相對' if a.relative else '絕對'}路徑）")
        if a.also_abs:
            make_kdenlive(shots, drafts, R(a.audio), out / "animatic_abs.kdenlive", n_frames, False)
            print(f"寫入 {out / 'animatic_abs.kdenlive'}（絕對路徑）")
    for k in [kd] + ([out / "animatic_abs.kdenlive"] if a.also_abs else []):
        if not a.no_kdenlive and k.exists():
            print(f"[{k.name}]", end=" ")
            verify_kdenlive(k, shots, n_frames)
    if not a.no_mp4 and mp4.exists():
        fails += verify_mp4(mp4, shots, n_frames, sl["meta"]["duration_sec"])
    if fails:
        for f in fails:
            print("驗證失敗：" + f, file=sys.stderr)
        sys.exit(1)
    print("驗證通過")


if __name__ == "__main__":
    main()
