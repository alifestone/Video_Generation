#!/usr/bin/env python3
"""P6 輔助：用 Qwen-Image-Edit 出單一物件／人物的素材圖（白底），存到 refs/p6/<name>_<seed>.png，之後由 tools/p6_keyframes.py 的 paste_asset 摳圖貼進場景。
~/creative/ComfyUI/comfyui/bin/python tools/p6_make_asset.py s012_child 1,2,3,4
素材的英文說明與 refs 寫在本檔 ASSETS。同 seed 重跑會覆蓋。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p6_keyframes as P

ASSETS = {
    # s012：追氣球的小孩側面奔跑立繪（refs/entities/balloon_child/back.png 只有背面；劇本要側身朝畫面右、char_left 腳離地、手伸向右上）
    "s012_child": dict(
        refs=["refs/entities/balloon_child/back.png"],
        qwen=("Image 1 is an unrelated character sheet: ignore it completely. Image 2 shows a small child in a yellow raincoat seen from behind. "
              "Draw this same child (the same yellow raincoat, the same dark blue rain boots, about 6 years old) in a pure side view in profile, "
              "facing the right side of the image, running, full body, large in the frame. The yellow hood is pulled far forward over the head so that the front brim of the hood completely covers the face: "
              "seen from the side you only see the yellow hood with the dark empty shadow inside its opening, and no eyes, no nose, no mouth, no cheek, no hair are visible. "
              "The foot nearest the viewer is on the ground and the other foot is lifted off the ground behind, "
              "one arm swinging back and the other arm stretched forward and up toward the upper right with the hand open. "
              "The background is plain pure white, with no ground, no shadow, no balloon, no other object. Clean Japanese anime cel shading with thin black line art, flat colors. No text."),
    ),
    # Prechorus s017–s020：懸浮車側視圖的換色版（refs/entities/hovercar/side.png 是珍珠白＋青色燈帶；劇本：第 2 台酒紅＋洋紅燈帶、第 3 台深藍＋青色燈帶、銀灰）
    **{f"hovercar_{name}": dict(
        refs=["refs/entities/hovercar/side.png"],
        qwen=("Image 1 is an unrelated character sheet: ignore it completely. Image 2 shows a futuristic wheel-less hovering capsule car in a flat side view. "
              "Draw exactly the same car with exactly the same shape, the same side view, the same size and position in the frame, the same dark windows and the same flat gently curved roof, "
              f"but the body shell is {shell} and the glowing underglow strip along the bottom is {glow}. "
              "The background is plain pure white, with no ground, no shadow ellipse below the car, no other object. "
              "Clean Japanese anime cel shading with thin black line art, flat colors. No text, no letters, no logos, no licence plate."),
    ) for name, shell, glow in (
        ("pearl", "a glossy pearl white paint", "bright cyan"),
        ("wine", "a deep wine red (burgundy) glossy paint", "bright magenta"),
        ("navy", "a deep dark navy blue glossy paint", "bright cyan"),
        ("silver", "a silver-grey metallic paint", "bright magenta"),
    )},
    # s020：懸浮車正面（車頭朝鏡頭，鏡頭略高於車頂）；先用前三台的顏色（遠處的車用同一張去飽和）
    **{f"hovercar_front_{name}": dict(
        refs=["refs/entities/hovercar/side.png", "refs/entities/hovercar/three_quarter.png"],
        qwen=("Image 1 is an unrelated character sheet: ignore it completely. Images 2 and 3 show a futuristic wheel-less hovering capsule car with dark windows and a glowing strip under the body. "
              "Draw exactly the same car in a straight FRONT view: the nose of the car points straight at the viewer, perfectly symmetrical, the camera is slightly above the roof so that the flat gently curved roof is visible above the windshield, "
              f"the body shell is {shell} and the glowing underglow strip along the bottom edge is {glow}, it hovers in the air with no wheels. "
              "The background is plain pure white, with no ground, no shadow ellipse, no other object. "
              "Clean Japanese anime cel shading with thin black line art, flat colors. No text, no letters, no logos, no licence plate."),
    ) for name, shell, glow in (
        ("pearl", "a glossy pearl white paint", "bright cyan"),
        ("wine", "a deep wine red (burgundy) glossy paint", "bright magenta"),
        ("navy", "a deep dark navy blue glossy paint", "bright cyan"),
    )},
    # s020：由 hovercar_front_navy_1.png（正面）換色
    **{f"hovercar_frontb_{name}": dict(
        refs=["refs/p6/hovercar_front_navy_1.png"],
        qwen=("Image 1 is an unrelated character sheet: ignore it completely. Image 2 shows a futuristic wheel-less hovering capsule car in a straight front view. "
              "Draw exactly the same picture with the same car in exactly the same straight front view, the same shape, the same size and position in the frame, "
              f"but the body shell is {shell} and the glowing underglow strip along the bottom edge is {glow}. "
              "The background is plain pure white, with no ground and no shadow. Clean Japanese anime cel shading with thin black line art, flat colors. No text, no letters, no logos."),
    ) for name, shell, glow in (
        ("pearl", "a glossy pearl white paint", "bright cyan"),
        ("wine", "a deep wine red (burgundy) glossy paint", "bright magenta"),
    )},
}

if __name__ == "__main__":
    name, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    a = ASSETS[name]
    out = P.ROOT / "refs/p6"
    out.mkdir(exist_ok=True)
    for s in seeds:
        (out / f"{name}_{s}.png").write_bytes(P.K.qwen(P.expand(a["qwen"], P.load_jobs().get("macros", {})), a["refs"], s))
        print(name, s, flush=True)
