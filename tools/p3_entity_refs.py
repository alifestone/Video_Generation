#!/usr/bin/env python3
"""P3 步驟 6：替反覆出現的非角色實體出固定參考圖，存到 refs/entities/<id>/。

python3 tools/p3_entity_refs.py [--only drone,pod] [--steps 4]

用 Qwen-Image-Edit-2511＋ Lightning 4 步：多數不給參考圖（＝文生圖）；規格第 6 欄給了來源圖的，就以那張為 image1 改圖
（ECHO 的其他角度從 front 改，因為文生圖會在光帶上多畫鏡頭或眼睛）。來源圖會複製到 ComfyUI 的 input/，經 ComfyUI HTTP API（127.0.0.1:8188）。
prompt 依 plan/glossary.yaml 各實體的 look 改寫成英文；完整 prompt 與 seed 記在 refs/entities/results.json。
"""
import argparse, json, shutil, time, urllib.request, urllib.parse
from pathlib import Path

HOST = "http://127.0.0.1:8188"
COMFY_INPUT = Path.home() / "creative/ComfyUI/input"
ROOT = Path(__file__).resolve().parent.parent
SEED0 = 20261025
STYLE = ("Japanese anime background art and prop design sheet style, clean thin line art, flat cel shading with 1-2 shadow steps, "
         "no text, no letters, no numbers, no logos, no watermark, no people unless stated. ")

DRONE = ("A spherical companion drone about 25 cm in diameter: a perfect sphere with a white shell (#f4f4f6) split into upper and lower halves "
         "by thin silver-grey seam lines, a horizontal glowing cyan light band (#29d3f5) on the front like a single eye, and a soft pale cyan glow "
         "ring under its bottom; no propellers, no arms, no face, no text. It floats in the air. ")
POD = ("A tall upright capsule pod, about 2.2 m tall and 1 m wide: lower part is a white metal base, upper part is a transparent curved glass "
       "canopy split into left and right halves; the round top surface of the base glows soft white (#e8f4ff), lighting upward. ")
TOWER = ("A slender broadcast tower about 240 m tall: a straight cylindrical shaft of matte black metal with six vertical structural ribs, "
         "each rib edged with a thin glowing cyan line; magenta glowing horizontal light bands ring the shaft every 12 m; at two thirds of the height "
         "a wide disc-shaped observation deck four times wider than the shaft with a ring of panoramic dark glass windows and an outer walkway "
         "with a low metal wall; above the deck a thin neck and a long thin antenna with a white warning light on top; not a lattice tower. ")
HOVERCAR = ("A near-future two-seat hover car: streamlined capsule body about 4.5 m long, flat roof, dark tinted windows, no wheels, "
            "a glowing light strip under the body, floating above the ground; no text or license plate. ")
CITY = "A near-future night city with tall buildings, holographic billboards showing only abstract colorful shapes without any text, light rain, wet reflective streets"

# (entity id, file name, width, height, prompt)
SPECS = [
    ("drone", "front", 1024, 1024, DRONE + "Front view, centered, plain white background, product design sheet."),
    ("drone", "three_quarter", 1024, 1024, "Rotate this same drone to a three-quarter view from the front left (turned about 45 degrees): the cyan band curves around the sphere; "
     "keep the same white shell, seam lines, plain cyan band and bottom glow exactly; plain white background.", "refs/entities/drone/front.png"),
    ("drone", "side", 1024, 1024, "Rotate this same drone to an exact side view (camera 90 degrees to its right): the glowing cyan band now curves around the right edge of the sphere and is seen edge-on; "
     "keep the same white shell, seam lines and bottom glow exactly; plain white background.", "refs/entities/drone/front.png"),
    ("drone", "back", 1024, 1024, "A spherical companion drone about 25 cm in diameter seen exactly from behind: a perfect smooth white sphere (#f4f4f6) with one thin silver-grey horizontal seam line around its middle and a soft pale cyan glow ring under its bottom; the back of the shell has NO light band, NO window, NO eye, NO markings, it is completely plain white; no propellers, no arms. It floats in the air. Plain white background."),
    ("drone", "night", 1024, 1024, "Keep this exact same drone unchanged (same shape, same plain cyan band, same bottom glow) and place it floating in a dark room at night; "
     "its cyan band and bottom glow softly light the air and the floor around it.", "refs/entities/drone/front.png"),
    ("pod", "closed", 832, 1216, POD + "Closed glass canopy, empty pod, front view, standing in the center of a dark round control room, plain dark background."),
    ("pod", "open", 832, 1216, "Edit this capsule pod into its open state, keeping the same base, size, colors and drawing style: the transparent glass canopy is split "
     "vertically down the middle into a left half and a right half; both halves have slid sideways away from each other and now stand beside the base on the left "
     "and on the right, each still a curved glass half-cylinder, leaving the middle completely open; the glowing white top of the base is exposed and empty; "
     "same dark room background, front view.", "refs/entities/pod/closed.png"),   # G0：照 glossary 從 closed 改圖
    ("pod", "platform", 1216, 832, "The round glowing white platform (1 m wide, 10 cm thick) on top of a white capsule pod base, seen from a high angle, dark room, soft white glow."),
    ("hq_control", "wide", 1344, 768, "An underground round control room about 12 m wide and 5 m high: the whole wall is a curved monitor wall of about 40 widescreen screens in three rows showing city views; "
     "in the center stands a tall white capsule pod with a glass canopy; black metal grating floor with thin cyan guide light lines converging toward the pod; no ceiling lights; cold blue-black palette; no people."),
    ("monitor_wall", "color", 1344, 768, "A curved monitor wall of about 40 widescreen screens in three rows inside a dark round room; every screen shows a different colorful night city view "
     "(neon streets, rooftops, plazas) without any text, user interface or numbers; the center screen of the middle row shows only a thin cyan waveform line on black."),
    ("monitor_wall", "frozen", 1344, 768, "A curved monitor wall of about 40 widescreen screens in three rows inside a dark round room; most screens show night city views in pure greyscale "
     "(frozen, no color), a few on the right still in color; no text, no user interface; the center screen of the middle row shows a thin cyan waveform line on black."),
    ("launch_tunnel", "shaft", 832, 1216, "Looking up a perfectly round vertical shaft 3 m wide made of matte black metal with thin vertical seams, cyan glowing rings (#29d3f5) "
     "Interior view from inside a dark vertical tunnel: the camera stands at the bottom and looks straight up, the black metal walls surround the viewer on all sides and converge "
     "toward a small round dark opening at the center of the image; glowing cyan rings (#29d3f5) on the inner wall every 6 m form concentric circles shrinking toward the center; "
     "the walls fill the whole frame, dark blue-black atmosphere."),
    ("launch_roof", "hatch", 1344, 768, "A flat concrete rooftop of a 150 m tall building at night, low parapet wall around the edge; in the center a round hatch 1.6 m wide flush with the floor, "
     "made of two semicircular steel plates, with a thin cyan glowing ring around it; beyond the parapet a sprawling night city; light rain. Not a helipad, no H mark."),
    ("street", "restored", 1344, 768, "A straight pedestrian shopping street 15 m wide at night, one-point perspective toward the end: glass shop windows on the ground floor without signs, "
     "holographic billboards with abstract images on the upper floors of 4-6 story buildings, wet stone pavement with puddles and no road markings, hover cars flying high in the air at second-floor height (6-10 m above the pavement, nothing on the ground), "
     "a slender tower silhouette far away at the end of the street, light rain; a few pedestrians seen from behind; no text anywhere."),
    ("street", "frozen", 1344, 768, "The same straight pedestrian shopping street at night in pure greyscale, completely frozen: raindrops hanging still in the air, hover cars stopped mid-air, "
     "pedestrians frozen like grey statues seen from behind, billboards and street lights still glowing but without color; far away at the end of the street a district still in color; no text."),
    ("hovercar", "side", 1344, 768, HOVERCAR + "Pearl white body with a cyan under-glow, side view, plain light grey background."),
    ("hovercar", "three_quarter", 1344, 768, HOVERCAR + "Deep blue body with a magenta under-glow, three-quarter view from the front, plain light grey background."),
    ("signal_tower", "full_night", 832, 1216, TOWER + "Full view from a distance at night above a city, all lights on, dark sky."),
    ("signal_tower", "deck", 1344, 768, "Close view of the disc-shaped observation deck of a black metal broadcast tower at night: a ring of panoramic dark glass windows 4 m tall with metal frames, "
     "a 1.5 m wide outer walkway cantilevered outside the windows with a 1 m low metal wall, magenta glowing bands along the top and bottom edges of the disc, "
     "below the deck six diagonal support struts with cyan glowing lines, city lights far below."),
    ("signal_tower", "base", 1344, 768, "The base of a black metal broadcast tower at night: a 10 m wide cylindrical shaft with six vertical ribs edged with cyan glowing lines, small grated maintenance platforms "
     "on the ribs every 8 m alternating left and right, magenta light bands ringing the shaft, a round plaza of wet stone around the base with puddles."),
    ("signal_tower", "control_offline", 1344, 768, "Inside a round broadcast control room 36 m wide with a 6 m ceiling: panoramic dark windows all around showing a night city, a round 4 m console in the center "
     "with a transparent glass light column rising to the ceiling, twelve radial rows of dark equipment cabinets, waist-high sub-consoles at both ends of each row; everything is switched off, "
     "the room is lit only by city light from the windows and a ring of dim white floor lights along the windows; no screens with text."),
    ("signal_tower", "control_online", 1344, 768, "The same round broadcast control room fully powered: in the exact center of the room a round 4 m console disc glows with cyan concentric rings and a magenta edge line, "
     "and a glass light column rises vertically from the very center of that disc (the rings surround the column's base) up to the ceiling, glowing bright cyan-white; "
     "cabinets have vertical cyan light strips and thin magenta top lines, every panoramic window shows a different colorful city image; no text, no numbers, no user interface."),
    ("balloon_child", "back", 832, 1216, "A six-year-old child seen from behind and slightly to the side, running along a wet street at night reaching one hand up toward a red helium balloon, "
     "wearing a yellow hooded raincoat with the hood up (#f2c230) and dark navy rain boots; the face is hidden by the hood; the balloon is round, glossy red (#d8323a) with a white string."),
]
CITY_VIEWS = [
    ("neon_street", "a narrow neon-lit street with crowds seen from behind"),
    ("rooftops", "rooftops and water tanks under drizzle with distant skyscrapers"),
    ("plaza", "a round plaza with a fountain and holographic billboards"),
    ("hover_lane", "hover cars without wheels flying high in the air between towers, several floors above the street, glowing under-body light strips, nothing touching the ground"),
    ("river", "a river embankment with reflections of city lights"),
    ("market", "a covered night market with lanterns of many colors, no signs"),
    ("station", "an elevated train station platform with a train arriving"),
]
for i, (name, desc) in enumerate(CITY_VIEWS, 1):
    SPECS.append(("monitor_wall", f"city_{i}_{name}", 1344, 768, f"{CITY}: {desc}. A single full-bleed widescreen illustration that fills the entire image edge to edge, no screen, no monitor frame, no bezel, no border, colorful, no text anywhere."))


def api(path, data=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def build(prompt, w, h, seed, steps):
    wf = json.loads((ROOT / "workflows/p0_ref_gen.json").read_text())
    for k in ("10", "11"):
        del wf[k]
    for k in ("20", "21"):
        wf[k]["inputs"].pop("image1", None)
        wf[k]["inputs"].pop("image2", None)
    wf["20"]["inputs"]["prompt"] = prompt
    wf["21"]["inputs"]["prompt"] = ""
    wf["6"] = {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["1", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}}
    wf["4"]["inputs"]["model"] = ["6", 0]
    wf["30"]["inputs"].update(width=w, height=h)
    wf["40"]["inputs"].update(seed=seed, steps=steps, cfg=1.0, sampler_name="euler", scheduler="simple")
    wf["60"]["inputs"]["filename_prefix"] = "p3ent/x"
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}
    # 兩輪都淘汰的規格（見 refs/entities/README.md）：不要因為檔案不存在又重出
    skip = {("drone", "three_quarter"), ("drone", "side"), ("drone", "night")}
    base = ROOT / "refs/entities"
    log_path = base / "results.json"
    recs = json.loads(log_path.read_text()) if log_path.exists() else []
    for idx, (eid, name, w, h, prompt, *src) in enumerate(SPECS, 1):
        if (only and eid not in only) or (eid, name) in skip:
            continue
        dest = base / eid / f"{name}.png"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        seed = SEED0 + idx
        full = prompt + " " + STYLE
        t0 = time.time()
        wf = build(full, w, h, seed, a.steps)
        if src:
            COMFY_INPUT.mkdir(exist_ok=True)
            img = f"p3ent_{eid}_src.png"
            shutil.copy(ROOT / src[0], COMFY_INPUT / img)
            wf["10"] = {"class_type": "LoadImage", "inputs": {"image": img}}
            for k in ("20", "21"):
                wf[k]["inputs"]["image1"] = ["10", 0]
        pid = api("/prompt", {"prompt": wf, "client_id": "p3ent"})["prompt_id"]
        while True:
            hist = api("/history/" + pid)
            if pid in hist and hist[pid].get("outputs"):
                break
            if pid in hist and hist[pid]["status"].get("status_str") == "error":
                raise RuntimeError(json.dumps(hist[pid]["status"])[:2000])
            time.sleep(2)
        im = hist[pid]["outputs"]["60"]["images"][0]
        dest.write_bytes(urllib.request.urlopen(f"{HOST}/view?" + urllib.parse.urlencode(im)).read())
        recs.append(dict(entity=eid, name=name, file=str(dest.relative_to(ROOT)), seed=seed, steps=a.steps, size=f"{w}x{h}",
                         seconds=round(time.time() - t0, 1), prompt=full, source=src[0] if src else None))
        log_path.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
        print(eid, name, recs[-1]["seconds"], flush=True)


if __name__ == "__main__":
    main()
