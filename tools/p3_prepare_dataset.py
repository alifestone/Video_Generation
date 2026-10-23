#!/usr/bin/env python3
"""P3 步驟 2–3：把挑選過的候選圖整理成 kohya sd-scripts 的資料集，並寫出訓練設定。

python3 tools/p3_prepare_dataset.py --selection lora/selection_v1.json --version v1

selection 檔是 {"keep": [{"file": "...png", "crop": [x0, y0, x1, y1] | null, "tags": "額外標籤", "note": "..."}], "reject": [...]}。
每張圖輸出到 lora/<version>/dataset/<repeats>_lunacsf/，旁邊放同名 .txt 標籤（Danbooru 風格，給 Illustrious）。
標籤只寫會變動的東西（構圖、角度、姿勢、表情、光線、背景）；角色固定外觀不寫，讓觸發詞 lunacsf 吸收。
"""
import argparse, json, re
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TRIGGER = "lunacsf"
BASE = "1girl, solo"

# 依 tools/p3_dataset_gen.py 的 SPECS 名稱給標籤
TAGS = {
    "fs_front_stand": "full body, standing, looking at viewer, arms at sides, white background, simple background",
    "fs_3q_left": "full body, standing, hand on hip, smile, three-quarter view, grey background, simple background",
    "fs_3q_right": "full body, standing, three-quarter view, confident, white background, simple background",
    "fs_side_left": "full body, walking, from side, profile, white background, simple background",
    "fs_side_right": "full body, standing, from side, profile, simple background",
    "fs_back": "full body, standing, from behind, white background, simple background",
    "fs_back_3q": "full body, from behind, looking back, looking at viewer, white background, simple background",
    "fs_low_angle": "full body, from below, clenched hands, determined, night sky, rim lighting",
    "fs_high_angle": "full body, from above, looking up, rooftop, night",
    "act_run": "full body, running, from side, motion lines, white background, simple background",
    "act_jump": "full body, jumping, midair, excited, simple background",
    "act_kneel_land": "full body, one knee, hand on ground, looking ahead, determined, rooftop, night",
    "act_crouch": "full body, crouching, hand on ground, looking up, indoors, dark, cyan light",
    "act_arms_open": "full body, outstretched arms, spread arms, smile, white background, simple background",
    "act_point": "cowboy shot, pointing at viewer, index finger raised, smile, white background, simple background",
    "act_reach_up": "cowboy shot, reaching up, looking up, dark background, light from above",
    "act_push_hands": "cowboy shot, hands forward, palms, determined, dark background",
    "act_fist_up": "cowboy shot, arm up, clenched hand, open mouth smile, night, city lights, bokeh",
    "act_walk_front": "full body, walking towards viewer, night, street, wet",
    "act_turn": "cowboy shot, from behind, looking back, white background, simple background",
    "ms_front_neutral": "upper body, looking at viewer, expressionless, white background, simple background",
    "ms_3q_smile": "upper body, smile, three-quarter view, simple background",
    "ms_3q_serious": "upper body, serious, three-quarter view, dark background, cyan light",
    "ms_side": "upper body, from side, profile, looking afar, night sky, rim lighting",
    "ms_back": "upper body, from behind, white background, simple background",
    "ms_hand_chest": "upper body, hand on own chest, looking at hand, white background, simple background",
    "ms_look_up": "upper body, looking up, parted lips, dark, light from above",
    "ms_laugh": "upper body, laughing, leaning forward, backlighting, sunrise",
    "mcu_front": "portrait, looking at viewer, determined, white background, simple background",
    "mcu_3q": "portrait, smile, three-quarter view, simple background",
    "mcu_surprised": "portrait, surprised, wide-eyed, white background, simple background",
    "mcu_side": "portrait, from side, profile, white background, simple background",
    "cu_face": "close-up, face, looking at viewer, earrings, white background, simple background",
    "cu_face_3q": "close-up, face, smile, three-quarter view, white background, simple background",
    "cu_eyes_closed": "close-up, face, closed eyes, dark background, light from below",
    "cu_eyes_open": "close-up, face, looking at viewer, dark background, light from below",
    "light_neon_fs": "full body, standing, looking to the side, night, city, neon lights, rim lighting",
    "light_neon_ms": "upper body, looking at viewer, smile, night, bokeh, rim lighting",
    "light_dark_room": "upper body, looking at viewer, indoors, dark, light from below, cyan light",
    "light_dawn_fs": "full body, from behind, looking afar, sunrise, horizon",
    "light_dawn_mcu": "portrait, looking at viewer, smile, sunrise, warm lighting",
    "light_rain": "upper body, looking at viewer, rain, night",
    "light_backlit": "full body, standing, looking at viewer, backlighting, white background",
    "light_overhead": "upper body, looking at viewer, serious, dark background, light from above",
    "bg_rooftop": "full body, standing, looking at viewer, rooftop, night, city lights",
    "bg_street": "full body, walking, three-quarter view, street, night, wet",
    "bg_control_room": "upper body, hands on table, looking at viewer, indoors, dark, cyan light",
    "bg_tower_walkway": "full body, standing, railing, looking afar, night, city lights, from above",
    "bg_pod": "full body, standing, closed eyes, glass capsule, dark, light from below",
    "bg_white_sit": "full body, sitting, knees up, hugging own legs, smile, white background, simple background",
    "expr_determined": "upper body, looking at viewer, determined, closed mouth, white background, simple background",
    "expr_playful": "upper body, looking at viewer, grin, one eye closed, white background, simple background",
    "expr_worried": "upper body, worried, looking at hand, three-quarter view, white background, simple background",
    "expr_calm_smile": "portrait, looking at viewer, smile, white background, simple background",
    "detail_legs_front": "lower body, thigh strap, legs, white background, simple background",
    "detail_legs_back": "lower body, from behind, thigh strap, legs, white background, simple background",
    "detail_hair_back": "from behind, close-up, hair, braid, hair chain, white background, simple background",
    "detail_top_front": "upper body, close-up, zipper, bracelet, white background, simple background",
    "fs_front_wide": "full body, standing, looking at viewer, wide shot, white background, simple background",
    "fs_dynamic_lean": "full body, leaning forward, ready to run, determined, white background, simple background",
}
CROP_TAGS = {"upper": "upper body", "cowboy": "cowboy shot", "portrait": "portrait", "face": "close-up, face"}


def spec_name(path):
    m = re.match(r"\d{3}_(.+)_s\d+\.png$", Path(path).name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    sel = json.loads((ROOT / a.selection).read_text())
    out = ROOT / "lora" / a.version / "dataset" / f"{a.repeats}_{TRIGGER}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        old.unlink()
    n = 0
    for i, item in enumerate(sel["keep"], 1):
        src = ROOT / item["file"]
        im = Image.open(src).convert("RGB")
        tags = item.get("tags_override") or TAGS.get(spec_name(src) or "", "")
        if item.get("crop"):
            im = im.crop(tuple(item["crop"]))
            kind = item.get("crop_kind")
            if kind:
                tags = CROP_TAGS[kind] + ", " + re.sub(r"^(full body|cowboy shot|upper body|portrait)(, )?", "", tags)
        if item.get("tags"):
            tags += ", " + item["tags"]
        stem = f"{i:03d}_{Path(item['file']).stem}"
        im.save(out / f"{stem}.png")
        (out / f"{stem}.txt").write_text(f"{TRIGGER}, {BASE}, {tags}\n")
        n += 1
    print(f"{n} 張 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
