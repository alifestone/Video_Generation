#!/usr/bin/env python3
"""把 plan/ 的結構化劇本轉成給人讀的 Markdown（docs/mv_plan.md）；線上 doc 也用這份內容。

python3 tools/render_plan.py [--plan plan] [--out docs/mv_plan.md]

@id 會換成名詞表裡的名稱（粗體）；時間位置會換成秒數。這份 Markdown 是產物，不要手改，要改就改 plan/。
"""
import argparse, re
from pathlib import Path

import yaml

from lint_plan import REF_RE, ROOT, Clock, load


FIELD_ZH = {"style": "風格", "role": "角色設定", "notes": "備註", "refs": "參考圖", "shape": "形狀", "size": "大小", "color": "顏色",
            "light": "光", "motion": "動態", "plot_functions": "劇情功能", "layout": "配置", "floor": "地面",
            "palette": "色調", "weather": "天氣", "variants": "版本", "extras": "路人", "opening": "開啟方式",
            "content": "內容", "off_state": "關機時", "segments": "分段", "count_rule": "倒數規則",
            "boundary": "邊界", "spread": "擴散", "realize": "做法", "platform": "平台",
            "bottom": "下端", "top": "上端", "wall": "牆面", "rings": "燈環", "building": "大樓", "roof": "樓頂",
            "hatch": "艙口", "state": "狀態", "silhouette": "剪影", "location": "位置", "state_at_hook": "Hook 時的狀態",
            "restore": "恢復", "duration": "持續時間", "return": "退回", "rain_drop": "雨滴",
            "travel": "移動方式", "timing": "時間", "position": "位置",
            "storefront": "店面", "ground": "路面", "air": "空中", "body": "體型", "outfit": "服裝",
            "pose_frozen": "定格姿勢", "face": "臉", "position_frozen": "定格位置",
            "footstep_patch": "腳步區塊", "surge": "湧升", "start": "開始", "progress": "進度"}
REALIZE_ZH = {"model": "影片模型生成", "post": "後製", "model+post": "影片模型＋後製", "none": "不動"}


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


class Renderer:
    def __init__(self, plan):
        self.plan = plan
        self.project = load(plan / "project.yaml")
        self.glossary = load(plan / "glossary.yaml")
        self.story = load(plan / "story.yaml")
        self.shots = load(plan / "shots.yaml").get("shots", [])
        tl = yaml.safe_load((ROOT / self.project["timeline"]).read_text(encoding="utf-8"))
        self.clock = Clock(tl)
        self.lyrics = {l["id"]: l["text"] for l in tl["lyrics"]}
        self.ents = self.glossary.get("entities", {})
        self.vocab = self.glossary.get("vocab", {})
        self.out = []

    def r(self, s):
        """把 @id 換成名稱，並移除換行（表格格內不能換行）。"""
        if s is None:
            return ""
        s = REF_RE.sub(lambda m: f"**{self.ents.get(m.group(1), {}).get('name', '@' + m.group(1))}**", str(s))
        s = " ".join(s.split())
        return re.sub(r"(?<=[，。；：、」）—])\s+|\s+(?=[，。；：、「（—])", "", s)

    def w(self, line=""):
        self.out.append(line)

    def pos(self, p):
        return f"bar {p[0]} 第 {p[1]} 拍" if p != [0, 0] else "歌曲開頭"

    def status(self, d):
        return "待確認" if d.get("status") == "proposed" else "已確認"

    def header(self):
        p = self.project
        self.w(f"# {p.get('title', 'MV')} 概念與劇本")
        self.w()
        self.w("> 由 `plan/` 的 YAML 以 `tools/render_plan.py` 產生，不要手改；修改請改 `plan/` 再重新產生。")
        self.w("> 「待確認」是 Claude 的提案，還沒經過使用者確認。")
        self.w()
        self.w("## 參數與已確認決定")
        self.w()
        self.w("| 項目 | 值 |")
        self.w("| --- | --- |")
        for k, v in {**p.get("params", {}), **p.get("decisions", {})}.items():
            if isinstance(v, (dict, list)):
                v = yaml.safe_dump(v, allow_unicode=True, default_flow_style=True).strip()
            self.w(f"| {k} | {self.r(v)} |")
        self.w()

    def story_section(self):
        s = self.story
        self.w("## 故事")
        self.w()
        self.w(f"**一句話**：{self.r(s.get('logline'))}")
        self.w()
        self.w(f"**主題**：{self.r(s.get('theme'))}　**基調**：{self.r(s.get('tone'))}　**結局**：{self.r(s.get('ending'))}")
        self.w()
        self.w("| 段落 | 小節 | 時間 | 發生什麼 | 情緒 | 歌詞對應 | 狀態 |")
        self.w("| --- | --- | --- | --- | --- | --- | --- |")
        for b in s.get("beats", []):
            sec = self.clock.sections.get(b["section"], {})
            t = f"{mmss(sec.get('t_start', 0))}–{mmss(sec.get('t_end', 0))}"
            self.w(f"| {b['section']} | {b['bars'][0]}–{b['bars'][1]} | {t} | {self.r(b.get('story'))} | "
                   f"{self.r(b.get('emotion'))} | {self.r(b.get('lyric_link'))} | {self.status(b)} |")
        self.w()

    def glossary_section(self):
        self.w("## 名詞表")
        self.w()
        self.w("劇本裡的粗體名稱都指這裡定義的東西。")
        self.w()
        for eid, e in self.ents.items():
            self.w(f"### {e.get('name')}（`@{eid}`，{e.get('kind')}，{self.status(e)}）")
            self.w()
            self.w(self.r(e.get("definition")))
            self.w()
            rows = []
            for k, v in (e.get("look") or {}).items():
                rows.append((k, v))
            for k in ("variants", "extras", "plot_functions", "role", "notes", "refs"):
                if e.get(k):
                    rows.append((k, e[k]))
            if rows:
                self.w("| 項目 | 內容 |")
                self.w("| --- | --- |")
                for k, v in rows:
                    if isinstance(v, dict):
                        v = "；".join(f"{a}：{b}" for a, b in v.items())
                    elif isinstance(v, list):
                        v = "；".join(str(x) for x in v)
                    self.w(f"| {FIELD_ZH.get(k, k)} | {self.r(v)} |")
                self.w()
            if e.get("not_this"):
                self.w("**不是**：" + "；".join(self.r(x) for x in e["not_this"]))
                self.w()

    def vocab_section(self):
        self.w("## 詞彙")
        self.w()
        for key, title in (("transition", "轉場"), ("post_effect", "後製效果"), ("variation", "變化量"), ("shot_size", "景別"),
                           ("camera_move", "運鏡"), ("camera_speed", "運鏡速度"), ("angle", "角度")):
            items = self.vocab.get(key) or {}
            self.w(f"**{title}**")
            self.w()
            self.w("| 值 | 意思 |")
            self.w("| --- | --- |")
            for k, v in items.items():
                self.w(f"| `{k}` | {self.r(v)} |")
            self.w()
        self.w("方向：畫面左右一律是觀眾看到的左右；角色自己的左右寫 char_left／char_right。")
        self.w()

    def shots_section(self):
        self.w("## 鏡頭")
        self.w()
        for sh in self.shots:
            sp = sh["span"]
            t0, t1 = self.clock.t(sp["from"]), self.clock.t(sp["to"])
            tr = sh.get("transition_in") or {}
            params = "、".join(f"{k}={v}" for k, v in tr.items() if k not in ("type", "note"))
            self.w(f"### {sh['id']}｜{self.pos(sp['from'])} → {self.pos(sp['to'])}（{mmss(t0)}–{mmss(t1)}，{t1 - t0:.2f} s）｜{self.status(sh)}")
            self.w()
            lyr = "；".join(f"{i}「{self.lyrics.get(i, '')}」" for i in sh.get("lyrics") or [])
            self.w("| 項目 | 內容 |")
            self.w("| --- | --- |")
            self.w(f"| 段落／歌詞 | {sh['section']}；{lyr or '—'} |")
            self.w(f"| 場景 | {self.r(sh.get('place'))} |")
            self.w(f"| 景別／角度 | `{sh.get('shot_size')}` {self.r(self.vocab['shot_size'].get(sh.get('shot_size')))}；`{sh.get('angle')}` |")
            self.w(f"| 路線／變化量 | route {sh.get('route')}；variation {sh.get('variation')} |")
            self.w(f"| 轉場（進入本鏡） | `{tr.get('type')}` {params} {self.r(tr.get('note'))} |")
            for sub in sh.get("subjects") or []:
                pl = sub.get("placement") or {}
                extra = "；".join(x for x in (sub.get("facing") and f"面向 {sub['facing']}", self.r(sub.get("state"))) if x)
                self.w(f"| 主體 {self.r(sub.get('ref'))} | 位置 {pl.get('x')}／{pl.get('y')}／{pl.get('depth')}；大小：{self.r(sub.get('size'))}"
                       + (f"；{extra}" if extra else "") + " |")
            self.w(f"| 首幀 | {self.r(sh.get('first_frame'))} |")
            mo = sh.get("motion") or {}
            cam, subj = mo.get("camera") or {}, mo.get("subject") or {}
            self.w(f"| 鏡頭運動 | `{cam.get('move')}` {cam.get('speed')}（{REALIZE_ZH.get(cam.get('realize'), cam.get('realize'))}） |")
            self.w(f"| 主體運動 | {self.r(subj.get('text')) or '—'}（{REALIZE_ZH.get(subj.get('realize'), subj.get('realize'))}） |")
            self.w(f"| 末幀 | {self.r(sh.get('last_frame')) or '—'} |")
            if sh.get("notes"):
                self.w(f"| 備註 | {self.r(sh['notes'])} |")
            if sh.get("post"):
                items = []
                for p in sh["post"]:
                    if isinstance(p, dict):
                        p = dict(p)
                        name = p.pop("effect", "")
                        if "at" in p:
                            p["at"] = self.pos(p["at"])
                        if "hold_until" in p:
                            p["hold_until"] = self.pos(p["hold_until"])
                        p = f"`{name}` " + "，".join(f"{k}={v}" for k, v in p.items())
                    items.append(str(p))
                self.w(f"| 後製 | {self.r('；'.join(items))} |")
            if sh.get("overlay"):
                ov = sh["overlay"]
                self.w(f"| 字卡 | 位置：{self.r(ov.get('zone'))}；內容：{self.r(ov.get('content'))} |")
            self.w()
            if sh.get("beat_events"):
                self.w("| 拍點 | 時間 | 對齊 | 發生什麼 |")
                self.w("| --- | --- | --- | --- |")
                for ev in sh["beat_events"]:
                    self.w(f"| {self.pos(ev['at'])} | {self.clock.t(ev['at']):.2f} s | {self.r(ev.get('sync')) or '—'} | {self.r(ev.get('event'))} |")
                self.w()

    def open_section(self):
        rows = []
        for eid, e in self.ents.items():
            rows += [(f"名詞 {e.get('name')}", q) for q in e.get("questions") or []]
        for b in self.story.get("beats", []):
            rows += [(f"故事 {b['section']}", q) for q in b.get("questions") or []]
        for sh in self.shots:
            rows += [(f"鏡頭 {sh['id']}", q) for q in sh.get("questions") or []]
        reviews = []
        for f in sorted((self.plan / "review").glob("*.yaml")):
            reviews += [it for it in load(f).get("issues") or [] if it.get("status") == "open"]
        if not rows and not reviews:
            return
        self.w("## 待回答的問題")
        self.w()
        if rows:
            self.w("**劇本裡標註的問題**（用 id 回答，例如「A01：……」）")
            self.w()
            self.w("| id | 位置 | 問題 |")
            self.w("| --- | --- | --- |")
            for i, (where, q) in enumerate(rows, 1):
                self.w(f"| A{i:02d} | {where} | {self.r(q)} |")
            self.w()
        if reviews:
            self.w("**歧義審稿提出的問題**（用 id 回答，例如「Q06：a」）")
            self.w()
            self.w("| id | 嚴重度 | 原文 | 問題 | 選項 |")
            self.w("| --- | --- | --- | --- | --- |")
            for it in reviews:
                opts = "　".join(f"{chr(97 + i)}. {self.r(o)}" for i, o in enumerate(it.get("options") or []))
                self.w(f"| {it['id']} | {it.get('severity')} | {self.r(it.get('quote'))} | "
                       f"{self.r(it.get('question'))} | {opts or '—'} |")
            self.w()

    def render(self):
        self.header()
        self.story_section()
        self.shots_section()
        self.glossary_section()
        self.vocab_section()
        self.open_section()
        return "\n".join(self.out).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=str(ROOT / "plan"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "mv_plan.md"))
    args = ap.parse_args()
    text = Renderer(Path(args.plan)).render()
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"寫入 {args.out}（{len(text.splitlines())} 行）")


if __name__ == "__main__":
    main()
