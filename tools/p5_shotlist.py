#!/usr/bin/env python3
"""P5 鏡頭清單：plan/shots.yaml ＋ analysis/timeline.json ＋ plan/p5_prompts.yaml → shotlist.json 與兩份工作清單。

python3 tools/p5_shotlist.py                      # 輸出 shotlist.json、edit/jobs_video.json、edit/jobs_post.json
python3 tools/p5_shotlist.py --check              # 只驗證與印摘要，不寫檔
python3 tools/p5_shotlist.py --prompts X.yaml --shotlist-out Y.json --jobs-dir DIR   # 測試用，不動正式檔

絕對幀規則（plan/README.md「時間位置」＋ timeline.meta.frame_rule）：
  frame = round(t * 24)；t 取自 timeline.beats 的 (小節, 拍) 時間，小數拍在相鄰兩拍的 t 之間線性內插，
  小節最後一拍的下一拍是下一小節第 1 拍（最後一小節則用 bars[-1].t_end）；[0,0] = 0；"end" = n_frames。
  不用鏡頭時長累加。驗證：第一鏡 start=0、相鄰鏡 end==下一鏡 start、最後一鏡 end==n_frames，任何一項失敗 exit 1。

p5_prompts.yaml 缺鏡頭時：該鏡 tier／model／keyframe／video 填 null 並警告，不失敗。
影片工作清單的生成長度：
  Wan2.2 16fps、單次最多 81 幀、幀數須為 4k+1；H3 24fps、單次上限取試作驗證過的 124 幀（pilot_report），幀數取 4 的倍數。
  handle 規則（Claude 代決）：關鍵幀是本鏡第一／最後一個完整畫面，有關鍵幀釘住的那一端不加 handle。
    頭端：首幀＝shot start，handle 0；尾端：mode=flf（有 keyframe.last）handle 0，mode=i2v（沒有 last）尾端多生 --tail-handle-sec（預設 0.4 秒）再裁掉。
    flf 的生成幀數取最接近的合法幀數（Wan 4k+1、H3 4 的倍數），差的部分靠整支 0.9–1.1 倍重定時補，retime.speed 記錄倍率。
  超過單次上限者拆成兩次生成：SPLITS 表（或 p5_prompts.yaml 的 split.at）指定拆點，沒指定的在最近中點的小節線自動拆；列在 job 的 segments。
"""
import argparse, hashlib, json, math, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FPS = 24

WAN_FPS, WAN_MAX = 16, 81          # Wan2.2 I2V：16fps，81 幀（4k+1）
H3_FPS, H3_MAX = 24, 124           # H3 FL2VA：24fps，試作用 124 幀（未驗證更長）
MIN_PER_RUN = {"wan_flf": 11.0, "h3_fl2va": 9.4}   # 單支分鐘數（pilot_report：Wan lightx2v 11、H3 turbo 9.4）
ATTEMPTS = (3, 4)                  # P7 每鏡平均 3–4 次
VIDEO_MODELS = ("wan_flf", "h3_fl2va")
ALL_MODELS = VIDEO_MODELS + ("parallax", "post_only", "none")


HANDLE_RULE = {
    "原則": "關鍵幀是本鏡第一／最後一個完整畫面（plan/README.md），有關鍵幀釘住的那一端不加 handle；對拍靠整支片段 0.9–1.1 倍變速或後製重定時，不靠 handle。",
    "頭端": "所有影片鏡頭 handle 0：first_frame（shots/sNNN/keyframe.png）＝shot start_frame，gen_index 0。",
    "尾端 flf": "mode=flf（有 keyframe.last）handle 0：last_frame 的 gen_index＝gen_frames-1，對應絕對幀 end_frame-1（end_frame 不含）。"
              "生成幀數取最接近的合法幀數，差的部分用 retime.speed（生成長度／需要長度，>1 表示播放加速）重定時成 n_frames，範圍 0.9–1.1。",
    "尾端 i2v": "mode=i2v（沒有 last，只給首幀）尾端多生 tail_handle_frames_24（0.4 s＝10 幀），剪輯時裁掉；gen_range_24 含這段，不變速。",
    "拆段": "超過單次上限（或劇本指定）的鏡頭拆成 segments（各段 flf）：mid 畫面是上段尾幀（幀 split-1）也是下段首幀（幀 split），接點兩幀幾乎相同，剪輯時刪一幀。",
    "mode 與 model": "model 維持 wan_flf／h3_fl2va；mode=flf 用 p4_combo 的首尾幀組合（wan22_lightx2v_flf／h3_turbo），mode=i2v 用只給首幀的組合（wan22_lightx2v／h3_turbo_i2v）。",
}


def rnd(x):
    return int(math.floor(x + 0.5))


class Clock:
    """(小節, 拍) → 絕對幀。"""

    def __init__(self, tl):
        self.n_frames = tl["meta"]["n_frames"]
        self.beats = {(b["bar"], b["beat_in_bar"]): b["t"] for b in tl["beats"]}
        self.bars = {b["bar"]: b for b in tl["bars"]}
        self.beats_in_bar = {}
        for (bar, k) in self.beats:
            self.beats_in_bar[bar] = max(self.beats_in_bar.get(bar, 0), k)
        self.last_bar = max(self.bars)
        self.end_t = self.bars[self.last_bar]["t_end"]

    def _t(self, bar, k):
        if (bar, k) in self.beats:
            return self.beats[(bar, k)]
        raise ValueError(f"timeline 沒有 [{bar},{k}]（小節 {bar} 只有 {self.beats_in_bar.get(bar, 0)} 拍）")

    def _next(self, bar, k):
        """(bar,k) 的下一拍時間。"""
        if (bar, k + 1) in self.beats:
            return self.beats[(bar, k + 1)]
        if (bar + 1, 1) in self.beats:
            return self.beats[(bar + 1, 1)]
        if bar == self.last_bar:
            return self.end_t
        raise ValueError(f"[{bar},{k}] 之後沒有下一拍可內插")

    def frame(self, pos):
        if pos == "end":
            return self.n_frames
        bar, beat = pos
        if bar == 0 and beat == 0:
            return 0
        k = int(math.floor(round(beat, 6)))
        frac = round(beat - k, 6)
        t0 = self._t(bar, k)
        t = t0 if frac == 0 else t0 + frac * (self._next(bar, k) - t0)
        return rnd(t * FPS)

    def t_of_frame(self, f):
        return round(f / FPS, 4)


def load_prompts(path):
    if not path.exists():
        return {}, {}
    d = yaml.safe_load(path.read_text()) or {}
    return d.get("defaults") or {}, d.get("shots") or {}


def gen_plan(model, n24, mode, tail_handle):
    """回傳 dict：gen_frames、gen_fps、gen_seconds、retime_speed、exceeds、tail_frames_24。
    flf：首尾幀釘在 shot 首末幀（相隔 (n-1)/24 秒），取最接近的合法生成幀數，retime_speed＝生成長度／需要長度（>1 表示播放時加速）。
    i2v：從首幀開始生成 n+tail_handle 幀的時間長度（ceil），不變速，尾端裁掉 handle。"""
    wan = model == "wan_flf"
    fps, cap = (WAN_FPS, WAN_MAX) if wan else (H3_FPS, H3_MAX)
    tail = tail_handle if mode == "i2v" else 0
    need = (n24 + tail - 1) / FPS                      # 首幀到尾端那一幀之間的秒數
    if wan:
        cands = list(range(5, cap + 1, 4))             # 4k+1
    else:
        cands = list(range(8, cap + 1, 4))             # 4 的倍數
    span = (lambda f: (f - 1) / fps)
    if mode == "flf":
        f = min(cands, key=lambda f: abs(span(f) / need - 1))          # 最接近的合法幀數
        if span(f) / need < 0.9 and f == cands[-1]:                    # 連單次上限都不夠長 → 超限
            f = None
    else:
        ok = [f for f in cands if span(f) >= need - 1e-9]
        f = ok[0] if ok else None
    sp = span(f) / need if mode == "flf" else 1.0
    return dict(gen_frames=f, gen_frames_needed=f, gen_fps=fps, gen_seconds=round(f / fps, 3), retime_speed=round(sp, 4),
                exceeds=False, tail_frames_24=tail)


def pos_s(p):
    return "end" if p == "end" else f"[{p[0]},{p[1]:g}]"


def build(shots_yaml, clock, defaults, prompts):
    warns, errs = [], []
    out = []
    for s in shots_yaml:
        sid = s["id"]
        try:
            f0 = clock.frame(s["span"]["from"])
            f1 = clock.frame(s["span"]["to"])
        except ValueError as e:
            errs.append(f"{sid}: {e}")
            continue
        cam, sub = s["motion"]["camera"], s["motion"]["subject"]

        def absf(p):
            try:
                return clock.frame(p)
            except ValueError as e:
                errs.append(f"{sid}: {e}")
                return None

        events = []
        for e in s.get("beat_events") or []:
            fr = absf(e["at"])
            ev = dict(at=e["at"], frame=fr, rel_frame=None if fr is None else fr - f0, sync=e.get("sync"), event=e["event"])
            if fr is not None and not (f0 <= fr < f1):
                warns.append(f"{sid}: beat_event {pos_s(e['at'])}（幀 {fr}）不在本鏡 [{f0},{f1}) 內")
            events.append(ev)
        post = []
        for p in s.get("post") or []:
            q = dict(p)
            if isinstance(p.get("at"), list):
                q["frame"] = absf(p["at"])
            if isinstance(p.get("hold_until"), list):
                q["hold_until_frame"] = absf(p["hold_until"])
            post.append(q)

        pr = prompts.get(sid)
        if pr is None:
            warns.append(f"{sid}: p5_prompts.yaml 沒有這一鏡，tier／model／prompt 填 null")
            pr = {}
        model = pr.get("model")
        if model is not None and model not in ALL_MODELS:
            errs.append(f"{sid}: model '{model}' 不在 {ALL_MODELS}")
        tier = pr.get("tier")
        if tier is not None and tier not in ("hero", "normal"):
            errs.append(f"{sid}: tier '{tier}' 必須是 hero 或 normal")
        kf = pr.get("keyframe") or {}
        entry = dict(
            id=sid, section=s["section"], story_beat=s.get("story_beat"),
            start_frame=f0, end_frame=f1, n_frames=f1 - f0,
            start_t=clock.t_of_frame(f0), end_t=clock.t_of_frame(f1),
            span=s["span"], lyrics=s.get("lyrics"), place=s.get("place"),
            shot_size=s["shot_size"], angle=s["angle"],
            camera=dict(move=cam["move"], speed=cam["speed"], realize=cam["realize"]),
            subject=dict(realize=sub["realize"], text=sub["text"].strip()),
            route=s["route"], variation=s["variation"],
            transition_in=s["transition_in"],
            tier=tier, model=model,
            keyframe=dict(first=kf.get("first"), last=kf.get("last")),
            video=pr.get("video"),
            prompt_notes=pr.get("notes"), split=pr.get("split"),
            plan_first_frame=s["first_frame"].strip(),
            plan_last_frame=(s["last_frame"] or "").strip() or None,
            beat_events=events, post=post, overlay=s.get("overlay"),
            status=s["status"],
            progress=dict(
                draft=(ROOT / "edit/drafts" / f"{sid}.png").exists(),
                keyframe=(ROOT / "shots" / sid / "keyframe.png").exists(),
                video=(ROOT / "shots" / sid / "video.mp4").exists()),
        )
        # 一致性檢查（只警告）
        if pr:
            if model in VIDEO_MODELS:
                if not entry["video"]:
                    warns.append(f"{sid}: model={model} 但 video prompt 是空的")
            if not kf.get("first") and model != "none":
                warns.append(f"{sid}: keyframe.first 是空的")
            needs_model = "model" in (cam["realize"], sub["realize"]) or "model+post" in (cam["realize"], sub["realize"])
            if needs_model and model not in VIDEO_MODELS:
                warns.append(f"{sid}: realize 含 model，但 model={model}")
            if model in VIDEO_MODELS and not needs_model:
                warns.append(f"{sid}: model={model}，但 camera／subject 的 realize 都不含 model")
            if model == "parallax" and cam["realize"] == "model":
                warns.append(f"{sid}: model=parallax，但 camera.realize=model")
            if s["variation"] in ("medium", "large") and not kf.get("last") and model in VIDEO_MODELS:
                warns.append(f"{sid}: variation={s['variation']} 但沒有末幀")
        out.append(entry)
    return out, warns, errs


def validate(out, n_frames):
    errs = []
    if not out:
        return ["沒有任何鏡頭"]
    if out[0]["start_frame"] != 0:
        errs.append(f"第一鏡 {out[0]['id']} start_frame={out[0]['start_frame']}，應為 0")
    for a, b in zip(out, out[1:]):
        if a["end_frame"] != b["start_frame"]:
            errs.append(f"{a['id']}.end={a['end_frame']} != {b['id']}.start={b['start_frame']}")
    if out[-1]["end_frame"] != n_frames:
        errs.append(f"最後一鏡 end_frame={out[-1]['end_frame']}，應為 n_frames={n_frames}")
    for s in out:
        if s["n_frames"] <= 0:
            errs.append(f"{s['id']}: n_frames={s['n_frames']} <= 0")
    if len({s["id"] for s in out}) != len(out):
        errs.append("鏡號重複")
    total = sum(s["n_frames"] for s in out)
    if total != n_frames:
        errs.append(f"鏡頭幀數總和 {total} != n_frames {n_frames}")
    return errs


# 超過單次上限、或劇本 notes 指定要拆的鏡頭：鏡號 → 拆點 [小節, 拍]、中間幀路徑、兩段 prompt 用途。
# 也可在 p5_prompts.yaml 的鏡頭下寫 split: {at: [8, 2]} 覆寫拆點。
SPLITS = {
    "s008": dict(at=[8, 2], reason="劇本 notes 指定從 [8,2] 拆（上：static 躍出落地；下：crane_up 揭示）",
                 prompt_use=["video prompt 前段：艙口打開、她躍出、單膝落地到站起（不含鏡頭升高那句）",
                             "video prompt 後段：站起後鏡頭升高約 5 m 並前推，停在她後上方俯視全城（不含艙口與躍出）"]),
    "s030": dict(at=[29, 1], reason="劇本指定從 [29,1] 拆",
                 prompt_use=["video prompt 前半：鏡頭緩升、她從塔下段蹬跳往上（前 4 拍）",
                             "video prompt 後半：繼續蹬跳到中段、遠方彩色街區被吞回（後 4 拍）"]),
}


def _segment(s, model, f0, f1, first, last, tail_handle, prompt_use, _unused=None):
    mode = "flf" if last else "i2v"
    n = f1 - f0
    g = gen_plan(model, n, mode, tail_handle)
    gf = g["gen_frames"]
    sp = g["retime_speed"] or 1.0
    seg = dict(
        start_frame=f0, end_frame=f1, n_frames=n, mode=mode,
        gen_fps=g["gen_fps"], gen_frames=gf, gen_frames_needed=g["gen_frames_needed"], gen_seconds=g["gen_seconds"],
        exceeds_single_run=g["exceeds"], retime=dict(speed=g["retime_speed"], within_0p9_1p1=(g["retime_speed"] is not None and 0.9 <= g["retime_speed"] <= 1.1)),
        handle_head_frames_24=0, handle_tail_frames_24=g["tail_frames_24"],
        gen_range_24=[f0, f1 + g["tail_frames_24"]],
        first_frame=dict(gen_index=0, frame=f0, **first),
        last_frame=(dict(gen_index=gf - 1, frame=f1 - 1, **last) if last else None),
        prompt_use=prompt_use,
        beat_events=[dict(at=e["at"], frame=e["frame"], gen_index=rnd((e["frame"] - f0) / FPS * g["gen_fps"] * sp), event=e["event"])
                     for e in s["beat_events"] if e["frame"] is not None and f0 <= e["frame"] < f1],
    )
    return seg


def video_jobs(out, tail_handle, clock):
    jobs, unassigned = [], []
    for s in out:
        need_model = "model" in s["camera"]["realize"] or "model" in s["subject"]["realize"]
        if s["model"] not in VIDEO_MODELS:
            if need_model and s["model"] is None:
                unassigned.append(dict(id=s["id"], reason="p5_prompts.yaml 沒有 model", start_frame=s["start_frame"], n_frames=s["n_frames"]))
            continue
        wan = s["model"] == "wan_flf"
        sid = s["id"]
        has_last = bool(s["keyframe"]["last"])
        first = dict(path=f"shots/{sid}/keyframe.png", prompt=s["keyframe"]["first"], exists=s["progress"]["keyframe"])
        last = dict(path=f"shots/{sid}/last.png", prompt=s["keyframe"]["last"]) if has_last else None
        mode = "flf" if has_last else "i2v"
        combo = ("wan22_lightx2v_flf" if mode == "flf" else "wan22_lightx2v") if wan else ("h3_turbo" if mode == "flf" else "h3_turbo_i2v")
        whole = _segment(s, s["model"], s["start_frame"], s["end_frame"], first, last, tail_handle, None, None)
        sp = SPLITS.get(sid, {})
        if s.get("split") and s["split"].get("at"):
            sp = dict(sp, at=s["split"]["at"], reason=sp.get("reason") or "p5_prompts.yaml 指定拆點")
        j = dict(
            id=sid, model=s["model"], mode=mode, tier=s["tier"], section=s["section"], p4_combo=combo,
            start_frame=s["start_frame"], end_frame=s["end_frame"], n_frames=s["n_frames"],
            video_prompt=s["video"], variation=s["variation"], route=s["route"], camera=s["camera"],
            subject_realize=s["subject"]["realize"],
        )
        if (whole["exceeds_single_run"] or sp) and not sp.get("at"):
            # 自動在最接近中點的小節線拆
            mid_t = (s["start_frame"] + s["end_frame"]) / 2
            bars = [(abs(clock.frame([b, 1]) - mid_t), b) for b in clock.bars
                    if s["start_frame"] < clock.frame([b, 1]) < s["end_frame"]]
            if bars:
                sp = dict(at=[min(bars)[1], 1], reason="超過單次上限，在最近中點的小節線自動拆", prompt_use=[None, None])
        if sp.get("at"):
            fm = clock.frame(sp["at"])
            if not (s["start_frame"] < fm < s["end_frame"]):
                raise SystemExit(f"{sid}: 拆點 {sp['at']}（幀 {fm}）不在本鏡內")
            mid = dict(path=f"shots/{sid}/mid.png", prompt=None, exists=(ROOT / "shots" / sid / "mid.png").exists(), needs_creation=True)
            segs = [
                dict(id=f"{sid}_a", **_segment(s, s["model"], s["start_frame"], fm, first, mid, tail_handle, sp["prompt_use"][0], None)),
                dict(id=f"{sid}_b", **_segment(s, s["model"], fm, s["end_frame"], dict(mid), last, tail_handle, sp["prompt_use"][1], None)),
            ]
            for sg in segs:
                sg["mode"] = "flf" if sg["last_frame"] else "i2v"
            j.update(split=dict(at=sp["at"], frame=fm, reason=sp["reason"], mid_keyframe=mid["path"],
                                single_run_would_fit=not whole["exceeds_single_run"]),
                     segments=segs, exceeds_single_run=any(x["exceeds_single_run"] for x in segs),
                     mode_note="每段都是 flf；拆點的 mid 畫面同時是上段尾幀（幀 split-1）與下段首幀（幀 split），接點有兩幀幾乎相同，剪輯時刪掉一幀")
            if j["exceeds_single_run"]:
                j["suggestion"] = "拆成兩段後仍有一段超過單次上限：再拆、縮短或請 P2 拆鏡"
        else:
            j.update(split=None, segments=None, exceeds_single_run=whole["exceeds_single_run"],
                     **{k: v for k, v in whole.items() if k not in ("prompt_use", "exceeds_single_run", "start_frame", "end_frame", "n_frames")})
            if whole["exceeds_single_run"]:
                j["suggestion"] = "超過單次上限且鏡內沒有小節線可拆：縮短、改視差／靜幀＋後製、或請 P2 拆鏡"
        jobs.append(j)
    return jobs, unassigned


def post_jobs(out):
    items = []

    def add(s, kind, **kw):
        items.append(dict(shot=s["id"], kind=kind, shot_start=s["start_frame"], shot_end=s["end_frame"], **kw))

    for s in out:
        cam, sub = s["camera"], s["subject"]
        if "post" in cam["realize"]:
            add(s, "camera", move=cam["move"], speed=cam["speed"], realize=cam["realize"],
                start_frame=s["start_frame"], end_frame=s["end_frame"], note="運鏡由後製做（push_in 等＝裁切縮放／視差）")
        if "post" in sub["realize"]:
            add(s, "subject", realize=sub["realize"], text=sub["text"], start_frame=s["start_frame"], end_frame=s["end_frame"])
        if s["model"] == "parallax":
            add(s, "parallax", start_frame=s["start_frame"], end_frame=s["end_frame"], keyframe=f"shots/{s['id']}/keyframe.png",
                tool="tools/p4_parallax.py", camera=cam, note="Depthflow 視差；沒有影片模型")
        if s["model"] == "post_only":
            add(s, "post_only", start_frame=s["start_frame"], end_frame=s["end_frame"], keyframe=f"shots/{s['id']}/keyframe.png",
                note="沒有影片模型，只用關鍵幀＋後製")
        for p in s["post"]:
            q = dict(p)
            eff = q.pop("effect")
            fr = q.pop("frame", None)
            add(s, "post_effect", effect=eff, start_frame=fr if fr is not None else s["start_frame"], params=q)
        for e in s["beat_events"]:
            if "post" in sub["realize"]:
                add(s, "beat_event", start_frame=e["frame"], at=e["at"], rel_frame=e["rel_frame"], event=e["event"], sync=e["sync"],
                    realize=sub["realize"])
        if s["transition_in"]["type"] != "cut":
            t = s["transition_in"]
            add(s, "transition_in", start_frame=s["start_frame"], type=t["type"], frames=t.get("frames"), note=t.get("note"))
        if s["overlay"]:
            ovs = s["overlay"] if isinstance(s["overlay"], list) else [s["overlay"]]
            for o in ovs:
                add(s, "overlay", start_frame=s["start_frame"], end_frame=s["end_frame"], overlay=o)
    items.sort(key=lambda i: (i["start_frame"], i["shot"]))
    return items


def runs_of(j):
    return len(j["segments"]) if j.get("segments") else 1


def summary(out, vjobs, unassigned, pitems, warns, tail_handle):
    n = len(out)
    cnt = {m: sum(1 for s in out if s["model"] == m) for m in ALL_MODELS}
    nomodel = sum(1 for s in out if s["model"] is None)
    print(f"\n=== P5 摘要：{n} 鏡，{out[-1]['end_frame']} 幀（{out[-1]['end_t']} s）===")
    print("各模型鏡數：" + "，".join(f"{m} {c}" for m, c in cnt.items()) + (f"，未指定 {nomodel}" if nomodel else ""))
    print(f"hero {sum(1 for s in out if s['tier'] == 'hero')}，normal {sum(1 for s in out if s['tier'] == 'normal')}，未指定 {sum(1 for s in out if s['tier'] is None)}")
    nruns = sum(runs_of(j) for j in vjobs)
    print(f"影片 job {len(vjobs)} 支（flf {sum(1 for j in vjobs if j['mode'] == 'flf')}、i2v {sum(1 for j in vjobs if j['mode'] == 'i2v')}），"
          f"拆段後生成單位 {nruns} 次（Wan {sum(runs_of(j) for j in vjobs if j['model'] == 'wan_flf')}、H3 {sum(runs_of(j) for j in vjobs if j['model'] == 'h3_fl2va')}）")
    hrs = lambda m, k: sum(runs_of(j) for j in vjobs if j["model"] == m) * MIN_PER_RUN[m] * k / 60
    for k in ATTEMPTS:
        print(f"預估生成工時（{k} 次/生成單位）：Wan {hrs('wan_flf', k):.1f} h ＋ H3 {hrs('h3_fl2va', k):.1f} h = {hrs('wan_flf', k) + hrs('h3_fl2va', k):.1f} h")
    mx = sum(MIN_PER_RUN[j["model"]] * runs_of(j) * (8 if j["tier"] == "hero" else 4) for j in vjobs) / 60
    print(f"預估上限（normal 最多 4 次、hero 最多 8 次，plan/p5_prompts.yaml 慣例）：{mx:.1f} h")
    print(f"（Wan {MIN_PER_RUN['wan_flf']:g} 分/支、H3 {MIN_PER_RUN['h3_fl2va']:g} 分/支；頭端 handle 0，i2v 尾端 handle {tail_handle} 幀）")
    sp = [j for j in vjobs if j["split"]]
    for j in sp:
        print(f"拆段：{j['id']}（{j['model']}）在 {j['split']['at']}（幀 {j['split']['frame']}）：" + "；".join(
            f"{g['id']} f{g['start_frame']}–{g['end_frame']} gen {g['gen_frames']}f 超限={g['exceeds_single_run']}" for g in j["segments"])
            + f"；{j['split']['reason']}；不拆也能單次完成={j['split']['single_run_would_fit']}")
    over = [j for j in vjobs if j["exceeds_single_run"]]
    print(f"拆段後仍超過單次上限：{len(over)} {', '.join(j['id'] for j in over)}")
    rts = [(j["id"], j["retime"]["speed"]) for j in vjobs if not j["segments"] and j["retime"]["speed"] and abs(j["retime"]["speed"] - 1) > 0.05]
    oor = [(j["id"], j["retime"]["speed"]) for j in vjobs if not j["segments"] and j["retime"]["speed"] and not j["retime"]["within_0p9_1p1"]]
    if oor:
        print(f"flf 重定時超出 0.9–1.1（鏡頭太短，Wan 幀數粒度 4 幀）：{len(oor)}：" + ", ".join(f"{i}×{v:.3f}" for i, v in oor))
    if rts:
        print(f"flf 需重定時超過 ±5% 的鏡頭：{len(rts)}：" + ", ".join(f"{i}×{v:.2f}" for i, v in rts[:15]))
    if unassigned:
        print(f"realize 含 model 但沒指定 model 的鏡頭：{len(unassigned)}")
    kinds = {}
    for i in pitems:
        kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    print("後製工作：" + "，".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    miss = [s["id"] for s in out if s["model"] is None]
    if miss:
        print(f"缺 prompt 的鏡頭 {len(miss)}：{', '.join(miss[:12])}{' …' if len(miss) > 12 else ''}")
    print(f"警告 {len(warns)} 則")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", default="plan/shots.yaml")
    ap.add_argument("--timeline", default="analysis/timeline.json")
    ap.add_argument("--prompts", default="plan/p5_prompts.yaml")
    ap.add_argument("--shotlist-out", default="shotlist.json")
    ap.add_argument("--jobs-dir", default="edit")
    ap.add_argument("--tail-handle-sec", type=float, default=0.4, help="i2v（沒有 keyframe.last）鏡頭尾端多生幾秒（P7 規格 0.3–0.5）")
    ap.add_argument("--check", action="store_true", help="只驗證與摘要，不寫檔")
    ap.add_argument("-q", "--quiet", action="store_true", help="不逐則印警告")
    a = ap.parse_args()

    R = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    tl = json.loads(R(a.timeline).read_text())
    shots = yaml.safe_load(R(a.shots).read_text())["shots"]
    clock = Clock(tl)
    if tl["meta"]["fps"] != FPS:
        sys.exit(f"timeline fps={tl['meta']['fps']}，本工具假設 {FPS}")
    defaults, prompts = load_prompts(R(a.prompts))
    if not prompts:
        print(f"警告：{a.prompts} 不存在或沒有 shots，所有鏡頭的 prompt 都是 null", file=sys.stderr)
    extra = sorted(set(prompts) - {s["id"] for s in shots})
    out, warns, errs = build(shots, clock, defaults, prompts)
    tail_handle = rnd(a.tail_handle_sec * FPS)
    for x in extra:
        warns.append(f"{x}: 在 p5_prompts.yaml 但不在 shots.yaml（忽略）")
    errs += validate(out, clock.n_frames) if not errs else []
    if not a.quiet:
        for w in warns:
            print("警告：" + w, file=sys.stderr)
    if errs:
        for e in errs:
            print("錯誤：" + e, file=sys.stderr)
        sys.exit(1)
    print(f"驗證通過：{len(out)} 鏡無縫覆蓋 0..{clock.n_frames}（相鄰 end==start）")

    vjobs, unassigned = video_jobs(out, tail_handle, clock)
    pitems = post_jobs(out)
    summary(out, vjobs, unassigned, pitems, warns, tail_handle)
    if a.check:
        return

    meta = dict(fps=FPS, n_frames=clock.n_frames, duration_sec=tl["meta"]["duration_sec"], n_shots=len(out),
                prompts_file=a.prompts, prompts_sha256=(hashlib.sha256(R(a.prompts).read_bytes()).hexdigest() if R(a.prompts).exists() else None),
                shots_sha256=hashlib.sha256(R(a.shots).read_bytes()).hexdigest(),
                timeline_sha256=hashlib.sha256(R(a.timeline).read_bytes()).hexdigest(),
                defaults=defaults, missing_prompts=[s["id"] for s in out if s["model"] is None],
                generated_by="tools/p5_shotlist.py")
    sl = R(a.shotlist_out)
    sl.write_text(json.dumps(dict(meta=meta, shots=out), ensure_ascii=False, indent=1) + "\n")
    jd = R(a.jobs_dir)
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "jobs_video.json").write_text(json.dumps(dict(
        meta=dict(fps=FPS, handle_rule=HANDLE_RULE, tail_handle_frames_24=tail_handle, wan=dict(fps=WAN_FPS, max_frames=WAN_MAX, rule="4k+1"),
                  h3=dict(fps=H3_FPS, max_frames=H3_MAX, rule="4 的倍數；124 為試作值，未驗證更長"),
                  min_per_run=MIN_PER_RUN, attempts=list(ATTEMPTS), defaults=defaults),
        jobs=vjobs, unassigned=unassigned), ensure_ascii=False, indent=1) + "\n")
    (jd / "jobs_post.json").write_text(json.dumps(dict(meta=dict(fps=FPS, n_frames=clock.n_frames), items=pitems), ensure_ascii=False, indent=1) + "\n")
    print(f"寫入 {sl}、{jd}/jobs_video.json、{jd}/jobs_post.json")


if __name__ == "__main__":
    main()
