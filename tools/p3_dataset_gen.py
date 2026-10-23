#!/usr/bin/env python3
"""P3 步驟 1：用 Qwen-Image-Edit-2511 從整張角色表擴充 LoRA 訓練候選圖。

python3 tools/p3_dataset_gen.py [--lightning] [--only 3,7,12] [--out lora/dataset_raw]

透過 ComfyUI HTTP API（127.0.0.1:8188）跑 workflows/p0_ref_gen.json（模式 A：只給整張角色表，見 refs/experiment/README.md 的決定）。
每一筆 SPECS 是一張圖：角度、景別、姿勢、表情、光線、背景各自變化，服裝固定。
--lightning 加 Qwen-Image-Edit-2511-Lightning-4steps LoRA（4 步、cfg 1），否則照 P0 的 35 步、cfg 4。
結果與完整 prompt 記在 <out>/results.json。
"""
import argparse, json, time, urllib.request, urllib.parse, uuid
from pathlib import Path

HOST = "http://127.0.0.1:8188"
ROOT = Path(__file__).resolve().parent.parent
SEED0 = 20261024

OUTFIT = ("Keep the exact same character and outfit as the reference sheet: same face with almond-shaped eyes and dark purple-grey irises, "
          "dark purple high ponytail with braids on the crown and a thin silver chain woven into the hair, "
          "white sleeveless high-collar zip-up crop top with a silver zipper from collar to hem, bare midriff, "
          "white cargo mini skirt with side pockets, black belt with a long black strap, one black thigh strap with a silver buckle on her right thigh only, "
          "several silver bracelets on both wrists, white ankle socks, white chunky sneakers; slim adult proportions; exactly five fingers on each hand. ")
STYLE = ("Single character only. Clean Japanese anime cel shading with thin even black line art and 1-2 shadow steps, "
         "same drawing style as the reference sheet. No text, no letters, no numbers, no logos, no captions, no symbols, no panel frames, no manga effect marks. ")
NEG = ""

# (name, width, height, view / framing / pose / expression, light and background)
SPECS = [
    # 全身：角度
    ("fs_front_stand", 704, 1024, "Full-body standing pose seen from the front, head to shoes visible, relaxed stance, arms at her sides, calm expression.", "plain white background, soft even studio light"),
    ("fs_3q_left", 704, 1024, "Full-body standing pose in three-quarter view turned toward the left of the image, head to shoes visible, one hand on her hip, slight smile.", "plain light grey background, soft studio light"),
    ("fs_3q_right", 704, 1024, "Full-body standing pose in three-quarter view turned toward the right of the image, head to shoes visible, arms relaxed, confident look.", "plain white background, soft studio light"),
    ("fs_side_left", 704, 1024, "Full-body side profile facing the left of the image, walking pose mid-stride, head to shoes visible.", "plain white background, soft studio light"),
    ("fs_side_right", 704, 1024, "Full-body side profile facing the right of the image, standing, looking ahead, head to shoes visible.", "plain pale blue-grey background, soft light"),
    ("fs_back", 704, 1024, "Full-body view from directly behind, head to shoes visible, the high ponytail and woven silver chain hanging down her back, the thigh strap on the right side of the image.", "plain white background, soft light"),
    ("fs_back_3q", 704, 1024, "Full-body view from behind in three-quarter, looking back over her shoulder toward the viewer, head to shoes visible.", "plain white background, soft light"),
    ("fs_low_angle", 704, 1024, "Full-body low-angle view looking up at her, standing heroically with fists clenched, determined expression, head to shoes visible.", "dark navy night sky background, cool cyan rim light from behind"),
    ("fs_high_angle", 704, 1024, "Full-body high-angle view looking down at her from above, she looks up toward the viewer, head to shoes visible.", "dark grey concrete rooftop floor, night, soft cool light"),
    # 全身：動作
    ("act_run", 1024, 768, "Full-body running pose seen from the side, dynamic stride, ponytail flowing behind her, focused expression.", "plain white background, soft light"),
    ("act_jump", 768, 1024, "Full-body mid-air jump, knees bent, arms swung back, ponytail lifted, excited expression.", "plain light background, soft light"),
    ("act_kneel_land", 1024, 768, "Full-body landing pose, one knee and one hand on the ground, looking up ahead with a determined expression.", "dark concrete rooftop at night, cool light"),
    ("act_crouch", 1024, 768, "Full-body low crouch with one hand touching the floor, looking up, alert expression.", "dark control-room floor, cool cyan light from the side"),
    ("act_arms_open", 768, 1024, "Full-body pose with both arms spread wide open, chest out, head slightly raised, joyful expression.", "plain white background, soft light"),
    ("act_point", 768, 1024, "Knee-up pose pointing forward toward the viewer with her right index finger, other fingers curled correctly, confident smile.", "plain white background, soft light"),
    ("act_reach_up", 768, 1024, "Knee-up pose reaching up with one open hand toward something above, looking up, hopeful expression.", "dark background with soft cyan light from above"),
    ("act_push_hands", 768, 1024, "Knee-up pose pressing both open palms forward against an invisible vertical surface at chest height, determined expression.", "dark background, cool light"),
    ("act_fist_up", 768, 1024, "Knee-up pose raising her right fist high into the air in victory, big open smile.", "night city background blurred, magenta and cyan neon bokeh"),
    ("act_walk_front", 704, 1024, "Full-body walking toward the viewer, front view, relaxed confident walk.", "wet night street blurred in background, neutral grey light"),
    ("act_turn", 768, 1024, "Knee-up pose turning around, body facing away but head turned back toward the viewer, ponytail swinging.", "plain white background, soft light"),
    # 中景／近景
    ("ms_front_neutral", 832, 1024, "Waist-up portrait facing the viewer, neutral calm expression, arms relaxed.", "plain white background, soft light"),
    ("ms_3q_smile", 832, 1024, "Waist-up portrait in three-quarter view toward the left of the image, gentle smile.", "plain light background, soft light"),
    ("ms_3q_serious", 832, 1024, "Waist-up portrait in three-quarter view toward the right of the image, serious focused expression.", "dark background, cool cyan key light from the right"),
    ("ms_side", 832, 1024, "Waist-up side profile facing the right of the image, looking into the distance.", "night sky background, soft cool rim light"),
    ("ms_back", 832, 1024, "Waist-up view from behind, ponytail and silver chain in the hair clearly visible.", "plain white background, soft light"),
    ("ms_hand_chest", 832, 1024, "Waist-up portrait with her right hand resting on her chest, eyes looking at her palm, curious expression.", "plain white background, soft light"),
    ("ms_look_up", 832, 1024, "Waist-up portrait looking up toward the upper left with wonder, lips slightly parted.", "dark room, soft cyan light from above"),
    ("ms_laugh", 832, 1024, "Waist-up portrait laughing with eyes open, joyful, slightly leaning forward.", "warm golden dawn light from behind, soft background"),
    ("mcu_front", 1024, 1024, "Chest-up close portrait facing the viewer, steady determined gaze.", "plain white background, soft light"),
    ("mcu_3q", 1024, 1024, "Chest-up close portrait in three-quarter view toward the left of the image, soft smile.", "plain light background, soft light"),
    ("mcu_surprised", 1024, 1024, "Chest-up close portrait, surprised expression with wide eyes and raised eyebrows.", "plain white background, soft light"),
    ("mcu_side", 1024, 1024, "Chest-up side profile facing the left of the image, calm.", "plain white background, soft light"),
    ("cu_face", 1024, 1024, "Close-up of her face from the front, eyes open and focused, neutral expression, earrings visible.", "plain white background, soft light"),
    ("cu_face_3q", 1024, 1024, "Close-up of her face in three-quarter view toward the right of the image, slight smile.", "plain white background, soft light"),
    ("cu_eyes_closed", 1024, 1024, "Close-up of her face with eyes gently closed, calm, as if just before waking.", "dark background, soft white light from below"),
    ("cu_eyes_open", 1024, 1024, "Close-up of her face with eyes just opened, focused gaze straight at the viewer.", "dark background, soft white light from below"),
    # 光線變化（夜景、控制室、黎明）
    ("light_neon_fs", 704, 1024, "Full-body standing pose in three-quarter view, looking off to the side.", "night city street with blurred neon signs without any text, magenta and cyan rim light, her white clothes stay white"),
    ("light_neon_ms", 832, 1024, "Waist-up portrait facing the viewer, confident smile.", "night city bokeh background without text, cyan rim light on one side and magenta on the other"),
    ("light_dark_room", 832, 1024, "Waist-up portrait facing the viewer, face lit from below by a cyan glow, focused.", "dark control room background, cyan light from below"),
    ("light_dawn_fs", 704, 1024, "Full-body standing pose seen from behind in three-quarter, looking at the horizon.", "dawn sky with warm golden light from the horizon, purple upper sky"),
    ("light_dawn_mcu", 1024, 1024, "Chest-up portrait facing the viewer, gentle smile, warm light on her face.", "warm golden dawn light, soft purple sky background"),
    ("light_rain", 832, 1024, "Waist-up portrait facing the viewer, rain drops falling around her, calm expression.", "rainy night background, cool blue-grey light"),
    ("light_backlit", 704, 1024, "Full-body standing pose facing the viewer, strong white light behind her creating a rim light, face still clearly lit.", "bright white background glow behind her"),
    ("light_overhead", 832, 1024, "Waist-up portrait facing the viewer, lit by a single soft light from directly above, serious expression.", "dark background"),
    # 背景場景（簡化，避免把背景學進 LoRA）
    ("bg_rooftop", 704, 1024, "Full-body standing pose facing the viewer near the edge of a flat rooftop, looking at the viewer.", "flat concrete rooftop at night, low parapet wall, distant city lights without text"),
    ("bg_street", 704, 1024, "Full-body pose walking along a street, three-quarter view toward the right of the image.", "wet stone pedestrian street at night with shop windows without any signs or text"),
    ("bg_control_room", 832, 1024, "Waist-up pose standing in front of a round console, hands resting on its edge, looking at the viewer.", "dark round control room with cyan console lights, no screens with text"),
    ("bg_tower_walkway", 704, 1024, "Full-body standing pose on a narrow metal walkway with a low metal wall, looking out over a city at night.", "high observation deck walkway at night, city lights far below"),
    ("bg_pod", 704, 1024, "Full-body standing pose inside a tall transparent glass capsule pod, eyes closed, arms relaxed.", "dark round room, white light from the floor of the pod below her"),
    ("bg_white_sit", 1024, 768, "Full-body sitting on the floor with knees up and arms around them, looking at the viewer with a soft smile.", "plain white background, soft light"),
    # 表情補充
    ("expr_determined", 832, 1024, "Waist-up portrait facing the viewer, determined expression with eyebrows slightly lowered, mouth closed.", "plain white background, soft light"),
    ("expr_playful", 832, 1024, "Waist-up portrait facing the viewer, playful grin, one eye winking.", "plain white background, soft light"),
    ("expr_worried", 832, 1024, "Waist-up portrait in three-quarter view, worried expression, looking at her hand held in front of her.", "plain white background, soft light"),
    ("expr_calm_smile", 1024, 1024, "Chest-up portrait facing the viewer, calm satisfied smile.", "plain white background, soft light"),
    # 腿環與服裝細節（不同朝向）
    ("detail_legs_front", 768, 1024, "Hip-to-shoes view from the front showing the white cargo mini skirt, the black thigh strap with silver buckle on her right thigh (on the left side of the image), white socks and white chunky sneakers.", "plain white background, soft light"),
    ("detail_legs_back", 768, 1024, "Hip-to-shoes view from behind showing the skirt, the black thigh strap on her right thigh (on the right side of the image), white chunky sneakers.", "plain white background, soft light"),
    ("detail_hair_back", 832, 1024, "View of her head and upper back from behind showing the braids on the crown and the silver chain woven into the high ponytail.", "plain white background, soft light"),
    ("detail_top_front", 832, 1024, "Chest and waist view from the front showing the white sleeveless high-collar crop top with the silver zipper running from collar to hem, silver bracelets on both wrists.", "plain white background, soft light"),
    ("fs_front_wide", 1024, 768, "Full-body standing pose facing the viewer, small in a wide landscape frame, arms relaxed.", "plain white background, soft light"),
    ("fs_dynamic_lean", 768, 1024, "Full-body pose leaning forward ready to sprint, one foot back, determined expression.", "plain white background, soft light"),
]


FRAMES = [
    ("Close-up of her face", "Framing: an extreme close crop; her face fills most of the image and the image is cut at her neck, nothing below the neck is visible. "),
    ("Chest-up", "Framing: a close portrait crop; the image is cut at the middle of her chest, nothing below the chest is visible. "),
    ("Waist-up", "Framing: a medium shot; the image is cut at her waist, her legs are not visible. "),
    ("Knee-up", "Framing: the image is cut just above her knees, her shoes are not visible. "),
    ("Hip-to-shoes", "Framing: only her body from the hips down to the shoes is visible; her head and chest are outside the image. "),
    ("head and upper back", "Framing: only her head and upper back are visible, seen from behind. "),
    ("Chest and waist", "Framing: only her torso from the neck to the hips is visible. "),
]


def framing(pose):
    for key, text in FRAMES:
        if key in pose:
            return text
    return "Framing: her whole body from head to shoes is inside the image. "


def api(path, data=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def upload(p):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{p.name}"\r\nContent-Type: image/png\r\n\r\n').encode() \
        + p.read_bytes() + (f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{b}--\r\n').encode()
    r = urllib.request.Request(HOST + "/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(r))["name"]


def build(spec, idx, sheet, lightning, seed):
    name, w, h, pose, light = spec
    wf = json.loads((ROOT / "workflows/p0_ref_gen.json").read_text())
    del wf["11"]
    del wf["20"]["inputs"]["image2"]
    del wf["21"]["inputs"]["image2"]
    wf["10"]["inputs"]["image"] = sheet
    prompt = ("Image 1 is a character reference sheet of the character. Create a new single illustration of this character. "
              + framing(pose) + pose + " Lighting and background: " + light + ". " + OUTFIT + STYLE)
    wf["20"]["inputs"]["prompt"] = prompt
    wf["21"]["inputs"]["prompt"] = NEG
    if framing(pose).startswith("Framing: her whole body") and max(w, h) <= 1024:
        w, h = (w * 1216 // 1024 // 16 * 16, 1216) if h >= w else (1216, h * 1216 // 1024 // 16 * 16)
    wf["30"]["inputs"].update(width=w, height=h)
    steps, cfg = (4, 1.0) if lightning else (35, 4.0)
    if lightning:
        wf["6"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                              "strength_model": 1.0}}
        wf["4"]["inputs"]["model"] = ["6", 0]
    wf["40"]["inputs"].update(seed=seed, steps=steps, cfg=cfg, sampler_name="euler", scheduler="simple")
    wf["60"]["inputs"]["filename_prefix"] = f"p3raw/{idx:03d}_{name}"
    return wf, prompt, steps, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lightning", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="lora/dataset_raw")
    ap.add_argument("--seed-offset", type=int, default=0, help="第二輪用不同 seed（例如 1000）")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "results.json"
    recs = json.loads(log_path.read_text()) if log_path.exists() else []
    only = {int(x) for x in a.only.split(",") if x}
    sheet = upload(ROOT / "refs/動畫_女生_LUNA.jpg")
    for idx, spec in enumerate(SPECS, 1):
        if only and idx not in only:
            continue
        seed = SEED0 + idx + a.seed_offset
        dest = out / f"{idx:03d}_{spec[0]}_s{a.seed_offset}.png"
        if dest.exists():
            continue
        wf, prompt, steps, cfg = build(spec, idx, sheet, a.lightning, seed)
        t0 = time.time()
        pid = api("/prompt", {"prompt": wf, "client_id": "p3"})["prompt_id"]
        while True:
            hist = api("/history/" + pid)
            if pid in hist and hist[pid].get("outputs"):
                break
            if pid in hist and hist[pid]["status"].get("status_str") == "error":
                raise RuntimeError(json.dumps(hist[pid]["status"])[:2000])
            time.sleep(2)
        im = hist[pid]["outputs"]["60"]["images"][0]
        dest.write_bytes(urllib.request.urlopen(f"{HOST}/view?" + urllib.parse.urlencode(im)).read())
        rec = dict(idx=idx, name=spec[0], file=str(dest.relative_to(ROOT)), seed=seed, steps=steps, cfg=cfg,
                   lightning=a.lightning, size=wf["30"]["inputs"]["width"].__str__() + "x" + str(wf["30"]["inputs"]["height"]), seconds=round(time.time() - t0, 1), prompt=prompt)
        recs.append(rec)
        log_path.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
        print(idx, spec[0], rec["seconds"], flush=True)


if __name__ == "__main__":
    main()
