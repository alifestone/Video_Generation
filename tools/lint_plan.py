#!/usr/bin/env python3
"""檢查 plan/ 的結構化劇本（glossary、story、shots）；有 FAIL 時結束碼為 1。

python3 tools/lint_plan.py [--plan plan] [--quiet]

檢查項目：
  G 名詞表   實體必填欄位、外觀是否定義
  R 引用     @id 是否存在、禁用別名與禁用母題、該用 @id 卻寫了名稱
  T 時間     span 是否落在拍點、鏡頭是否首尾相接、長度上限、beat_events 是否在鏡頭內
  S 鏡頭     必填欄位、詞彙值、主體位置／大小／面向、末幀、轉場參數、route 限制
  W 寫法     首幀寫了動作、模糊詞、「閃三次」這類沒拆成 beat_events 的序列
  N 敘事     段落是否齊全、敘事層混入鏡頭用語
最後列出所有 status: proposed 的項目與 questions。
"""
import argparse, re, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REF_RE = re.compile(r"@([a-z_][a-z0-9_]*)")
KINDS = {"character", "place", "object", "vfx", "state"}
ROUTES = {"A", "B", "C"}
REALIZE = {"model", "post", "model+post", "none"}
CLOSE = {"ECU", "CU"}
MOTION_WORDS = ["正在", "開始", "逐漸", "慢慢", "轉向", "走向", "即將", "接著", "然後", "之後", "滑入", "飛過"]
VAGUE_WORDS = ["一些", "某種", "某個", "適當", "適度", "若干", "等等", "之類", "很多", "許多", "幾個", "一點",
               "氛圍感", "感覺", "似乎", "大概", "好像", "閃幾下", "連閃"]
SEQ_RE = re.compile(r"(閃|亮|跳)(兩|三|四|五|\d)(次|下)|連(閃|亮)")
STORY_CAMERA_WORDS = ["特寫", "近景", "中景", "遠景", "全身", "推近", "拉遠", "鏡頭", "構圖", "畫面左", "畫面右", "景別"]
TRANSITION_PARAMS = {"flash_white": [("frames",)], "whiteout": [("in_beats",), ("out_beats",)],
                     "fade_from_black": [("beats", "frames")], "fade_to_black": [("beats", "frames")],
                     "dissolve": [("beats", "frames")]}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, code, where, msg):
        self.rows.append((level, code, where, msg))

    def fail(self, *a): self.add("FAIL", *a)
    def warn(self, *a): self.add("WARN", *a)


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def texts(node, path=""):
    """逐一產生 (路徑, 字串)；跳過 questions（問題本身可以提到任何詞）。"""
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            if k != "questions":
                yield from texts(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from texts(v, f"{path}[{i}]")


class Clock:
    """把 [小節, 拍] 換成秒；拍可為小數，在相鄰兩拍之間內插。"""

    def __init__(self, tl):
        self.duration = tl["meta"]["duration_sec"]
        self.bar_sec = tl["meta"]["bar_sec"]
        self.beats = sorted(tl["beats"], key=lambda b: b["t"])
        self.index = {(b["bar"], b["beat_in_bar"]): i for i, b in enumerate(self.beats)}
        self.bars = {b["bar"]: b for b in tl["bars"]}
        self.sections = {s["label"]: s for s in tl["sections"]}

    def t(self, pos):
        if pos == "end":
            return self.duration
        if not (isinstance(pos, list) and len(pos) == 2):
            raise ValueError(f"位置格式應為 [小節, 拍] 或 end：{pos!r}")
        bar, beat = pos
        if [bar, beat] == [0, 0]:
            return 0.0
        whole = int(beat)
        if (bar, whole) not in self.index:
            raise ValueError(f"[{bar},{beat}] 不是 timeline 裡的拍點")
        i = self.index[(bar, whole)]
        frac = beat - whole
        if frac == 0:
            return self.beats[i]["t"]
        nxt = self.beats[i + 1]["t"] if i + 1 < len(self.beats) else self.duration
        return self.beats[i]["t"] + frac * (nxt - self.beats[i]["t"])


def check_glossary(g, rep):
    vocab, ents = g.get("vocab", {}), g.get("entities", {})
    for eid, e in ents.items():
        w = f"glossary.{eid}"
        for f in ("kind", "name", "status", "definition"):
            if not e.get(f):
                rep.fail("G1", w, f"缺少 {f}")
        if e.get("kind") not in KINDS:
            rep.fail("G1", w, f"kind 應為 {sorted(KINDS)}")
        if e.get("kind") != "character" and not e.get("look"):
            rep.warn("G2", w, "外觀（look）尚未定義；寫到用它的鏡頭前要補上")
        if e.get("kind") == "character" and not (e.get("look") or e.get("refs")):
            rep.warn("G2", w, "角色沒有 look 也沒有參考圖 refs")
    return vocab, ents


def check_refs(docs, ents, banned, rep):
    names = {eid: e.get("name", "") for eid, e in ents.items()}
    aliases = {a: eid for eid, e in ents.items() for a in e.get("aliases_forbidden", []) or []}
    for doc_name, doc in docs:
        for path, s in texts(doc):
            where = f"{doc_name}:{path}"
            for rid in REF_RE.findall(s):
                if rid not in ents:
                    rep.fail("R1", where, f"@{rid} 不在名詞表")
            for a, eid in aliases.items():
                if a in s:
                    rep.fail("R2", where, f"用了禁用別名「{a}」，應寫 @{eid}")
            for b in banned:
                if b in s:
                    rep.fail("R2", where, f"出現已禁用的母題「{b}」")
            if doc_name == "glossary":
                continue
            plain = REF_RE.sub("", s)
            for eid, name in names.items():
                core = re.sub(r"（.*?）", "", name)
                if len(core) >= 2 and core in plain:
                    rep.warn("R3", where, f"寫了名稱「{core}」，應改成 @{eid}")


def check_writing(doc_name, doc, rep):
    for path, s in texts(doc):
        where = f"{doc_name}:{path}"
        hits = [w for w in VAGUE_WORDS if w in s]
        if hits:
            rep.warn("W2", where, f"模糊詞：{'、'.join(hits)}")
        if SEQ_RE.search(s) and "beat_events" not in path:
            rep.warn("W3", where, f"「{SEQ_RE.search(s).group(0)}」這類序列要拆成 beat_events，一拍一筆")


def check_raw_comments(plan, rep):
    """YAML 的「空格＋#」會把後面當成註解吃掉（例如色碼 #29d3f5）；值裡有 # 就要加引號。"""
    pat = re.compile(r"^\s*(?:- )?[a-z_]+: ([^\"'\[{>|\s#][^#]*?)\s#\S")
    for name in ("glossary", "story", "shots"):
        for i, line in enumerate((plan / f"{name}.yaml").read_text(encoding="utf-8").splitlines(), 1):
            if pat.match(line):
                rep.fail("Y1", f"{name}.yaml:{i}", "值裡的「 #」之後會被 YAML 當成註解刪掉，請把整個值加上引號")


def check_story(story, clock, rep):
    beats = story.get("beats", [])
    seen = {b.get("section") for b in beats}
    for label, sec in clock.sections.items():
        if label not in seen:
            rep.fail("N1", "story", f"段落 {label}（bar {sec['bar_start']}–{sec['bar_end']}）沒有敘事")
    for b in beats:
        w = f"story.{b.get('id')}"
        for f in ("id", "section", "bars", "story", "emotion", "lyric_link", "status"):
            if not b.get(f):
                rep.fail("N2", w, f"缺少 {f}")
        sec = clock.sections.get(b.get("section"))
        if sec and b.get("bars") != [sec["bar_start"], sec["bar_end"]]:
            rep.fail("N2", w, f"bars 應為 [{sec['bar_start']}, {sec['bar_end']}]（timeline）")
        hits = [x for x in STORY_CAMERA_WORDS if x in (b.get("story") or "")]
        if hits:
            rep.warn("N3", w, f"敘事層混入鏡頭用語：{'、'.join(hits)}（畫面細節寫在 shots）")
    return {b.get("id") for b in beats}


def bars_spanned(span):
    """鏡頭碰到幾個小節（弱起的 bar 0 也算一個小節）；to 落在下一小節第 1 拍時不算進下一小節。"""
    a, b = span["from"], span["to"]
    if b == "end":
        return 1
    return (b[0] if b[1] == 1 else b[0] + 1) - a[0]


def need(d, keys, where, rep, code="S1"):
    for k in keys:
        if d.get(k) in (None, "", []):
            rep.fail(code, where, f"缺少 {k}")


def check_shots(shots, vocab, ents, story_ids, clock, tl, params, rep):
    lyric_ids = {l["id"] for l in tl["lyrics"]}
    prev = None
    for sh in shots:
        sid = sh.get("id", "?")
        w = f"shots.{sid}"
        need(sh, ["id", "section", "story_beat", "span", "place", "shot_size", "angle", "route", "variation",
                  "subjects", "first_frame", "motion", "transition_in", "status"], w, rep)
        # 詞彙值
        for key, voc in (("shot_size", "shot_size"), ("angle", "angle")):
            if sh.get(key) and sh[key] not in vocab.get(voc, {}):
                rep.fail("S2", w, f"{key}={sh[key]} 不在 vocab.{voc}")
        if sh.get("route") not in ROUTES:
            rep.fail("S2", w, f"route 應為 {sorted(ROUTES)}")
        if sh.get("variation") not in vocab.get("variation", {}):
            rep.fail("S2", w, f"variation={sh.get('variation')} 不在 vocab.variation")
        if sh.get("story_beat") and sh["story_beat"] not in story_ids:
            rep.fail("S2", w, f"story_beat={sh['story_beat']} 不在 story.yaml")
        for lid in sh.get("lyrics") or []:
            if lid not in lyric_ids:
                rep.fail("S2", w, f"歌詞 id {lid} 不存在")
        # 主體
        for i, sub in enumerate(sh.get("subjects") or []):
            sw = f"{w}.subjects[{i}]"
            need(sub, ["ref", "placement", "size"], sw, rep, "S3")
            rid = (REF_RE.findall(sub.get("ref", "")) or [None])[0]
            pl = sub.get("placement") or {}
            for axis, voc in (("x", "screen_x"), ("y", "screen_y"), ("depth", "depth")):
                if pl.get(axis) not in vocab.get(voc, []):
                    rep.fail("S3", sw, f"placement.{axis}={pl.get(axis)} 不在 vocab.{voc}")
            if rid in ents and ents[rid].get("kind") == "character":
                need(sub, ["facing", "state"], sw, rep, "S3")
                if sub.get("facing") and sub["facing"] not in vocab.get("facing", []):
                    rep.fail("S3", sw, f"facing={sub['facing']} 不在 vocab.facing")
        # 畫面裡寫到的實體，都要列在 subjects 或 place（才有位置與大小）
        listed = {(REF_RE.findall(x.get("ref", "")) or [None])[0] for x in sh.get("subjects") or []}
        listed |= set(REF_RE.findall(sh.get("place") or ""))
        body = {k: sh.get(k) for k in ("first_frame", "last_frame", "motion", "beat_events")}
        shown = {rid for _, t in texts(body) for rid in REF_RE.findall(t)}
        for rid in sorted(shown - listed):
            if ents.get(rid, {}).get("kind") not in ("place", "state"):
                rep.warn("S7", w, f"畫面裡寫到 @{rid}，但 subjects 沒有列出它的位置與大小")
        # 運動
        mo = sh.get("motion") or {}
        cam, subj = mo.get("camera") or {}, mo.get("subject") or {}
        if cam.get("move") not in vocab.get("camera_move", {}):
            rep.fail("S4", w, f"motion.camera.move={cam.get('move')} 不在 vocab.camera_move")
        if cam.get("speed") not in vocab.get("camera_speed", {}):
            rep.fail("S4", w, f"motion.camera.speed={cam.get('speed')} 不在 vocab.camera_speed")
        for part, d in (("camera", cam), ("subject", subj)):
            if d.get("realize") not in REALIZE:
                rep.fail("S4", w, f"motion.{part}.realize 應為 {sorted(REALIZE)}")
        if sh.get("variation") in ("medium", "large") and not sh.get("last_frame"):
            rep.fail("S4", w, f"variation={sh.get('variation')} 必須寫 last_frame")
        # 首幀不寫動作
        hits = [x for x in MOTION_WORDS if x in (sh.get("first_frame") or "")]
        if hits:
            rep.warn("W1", w, f"first_frame 應是靜止畫面，出現動作詞：{'、'.join(hits)}")
        # 轉場
        tr = sh.get("transition_in") or {}
        ttype = tr.get("type")
        if ttype not in vocab.get("transition", {}):
            rep.fail("S5", w, f"transition_in.type={ttype} 不在 vocab.transition")
        for alts in TRANSITION_PARAMS.get(ttype, []):
            if not any(tr.get(a) for a in alts):
                rep.fail("S5", w, f"{ttype} 需要參數 {' 或 '.join(alts)}")
        if ttype == "match_cut" and not tr.get("note"):
            rep.fail("S5", w, "match_cut 要在 note 寫出前後對齊的是什麼")
        # 時間
        span = sh.get("span") or {}
        try:
            t0, t1 = clock.t(span.get("from")), clock.t(span.get("to"))
        except (ValueError, TypeError) as e:
            rep.fail("T1", w, str(e))
            prev = sh
            continue
        dur = t1 - t0
        if dur <= 0:
            rep.fail("T1", w, "span.to 必須晚於 span.from")
        nbars = bars_spanned(span)
        if nbars > params.get("max_shot_bars", 2):
            rep.fail("T2", w, f"跨 {nbars} 小節（{dur:.2f} s），超過 {params.get('max_shot_bars', 2)} 小節上限")
        if prev is None and span.get("from") != [0, 0]:
            rep.fail("T3", w, "第一鏡應從 [0, 0] 開始")
        if prev is not None and (prev.get("span") or {}).get("to") != span.get("from"):
            rep.fail("T3", w, f"與上一鏡 {prev.get('id')} 沒有首尾相接（{(prev.get('span') or {}).get('to')} ≠ {span.get('from')}）")
        sec = clock.sections.get(sh.get("section"))
        bar0 = span["from"][0] if isinstance(span.get("from"), list) else None
        if sec and bar0 is not None and not (sec["bar_start"] <= bar0 <= sec["bar_end"]):
            rep.fail("T4", w, f"起點 bar {bar0} 不在段落 {sh['section']}（bar {sec['bar_start']}–{sec['bar_end']}）內")
        for j, ev in enumerate(sh.get("beat_events") or []):
            try:
                te = clock.t(ev.get("at"))
            except (ValueError, TypeError) as e:
                rep.fail("T5", f"{w}.beat_events[{j}]", str(e))
                continue
            if not (t0 <= te < t1):
                rep.fail("T5", f"{w}.beat_events[{j}]", f"{ev.get('at')} 不在本鏡時間內")
            if not ev.get("event"):
                rep.fail("T5", f"{w}.beat_events[{j}]", "缺少 event")
        # route 限制（見計畫書「影片路線與模型」）
        if sh.get("route") == "C":
            if nbars > 1:
                rep.fail("S6", w, "route C 鏡頭不得超過 1 小節")
            if sh.get("shot_size") in CLOSE:
                rep.fail("S6", w, "route C（視差）不用於特寫")
            if subj.get("realize") in ("model", "model+post"):
                rep.fail("S6", w, "route C 是靜態圖＋視差，主體動作只能由後製做（realize: post）")
        prev = sh
    if prev is not None:
        try:
            end = clock.t((prev.get("span") or {}).get("to"))
            rep.add("INFO", "T6", "shots", f"鏡頭覆蓋到 {end:.2f} s／全曲 {clock.duration:.2f} s")
        except (ValueError, TypeError):
            pass


def collect_reviews(plan, rep):
    for f in sorted((plan / "review").glob("*.yaml")):
        for it in (load(f).get("issues") or []):
            if it.get("status") == "open":
                rep.add("OPEN", "Q2", f"{f.stem}.{it.get('id')} {it.get('where')}", it.get("question"))


def collect_open(docs, rep):
    for doc_name, doc in docs:
        items = []
        if doc_name == "glossary":
            items = [(f"glossary.{k}", v) for k, v in (doc.get("entities") or {}).items()]
        elif doc_name == "story":
            items = [("story", doc)] + [(f"story.{b.get('id')}", b) for b in doc.get("beats", [])]
        elif doc_name == "shots":
            items = [(f"shots.{s.get('id')}", s) for s in doc.get("shots", [])]
        for where, d in items:
            if d.get("status") == "proposed":
                rep.add("OPEN", "P1", where, "status: proposed（待使用者確認）")
            for q in d.get("questions") or []:
                rep.add("OPEN", "Q1", where, q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=str(ROOT / "plan"))
    ap.add_argument("--quiet", action="store_true", help="不列 OPEN 項目")
    args = ap.parse_args()
    plan = Path(args.plan)

    project = load(plan / "project.yaml")
    glossary, story, shots = (load(plan / f"{n}.yaml") for n in ("glossary", "story", "shots"))
    tl = yaml.safe_load((ROOT / project["timeline"]).read_text(encoding="utf-8"))
    clock = Clock(tl)
    params = project.get("params", {})
    banned = (project.get("decisions") or {}).get("banned_motifs", [])

    rep = Report()
    check_raw_comments(plan, rep)
    vocab, ents = check_glossary(glossary, rep)
    docs = [("glossary", glossary), ("story", story), ("shots", shots)]
    check_refs(docs, ents, banned, rep)
    for name, doc in docs[1:]:
        check_writing(name, doc, rep)
    story_ids = check_story(story, clock, rep)
    check_shots(shots.get("shots", []), vocab, ents, story_ids, clock, tl, params, rep)
    if not args.quiet:
        collect_open(docs, rep)
        collect_reviews(plan, rep)

    order = {"FAIL": 0, "WARN": 1, "INFO": 2, "OPEN": 3}
    for level, code, where, msg in sorted(rep.rows, key=lambda r: order[r[0]]):
        print(f"{level:4} {code} {where}: {msg}")
    n = {k: sum(r[0] == k for r in rep.rows) for k in order}
    print(f"\nFAIL {n['FAIL']}  WARN {n['WARN']}  OPEN {n['OPEN']}")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
