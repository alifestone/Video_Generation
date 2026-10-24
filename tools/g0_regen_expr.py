#!/usr/bin/env python3
"""G0（2026-10-24 使用者選）：重出 refs/04、05 表情參考圖——胸口以上、睜眼。

~/creative/ComfyUI/comfyui/bin/python tools/g0_regen_expr.py [--seeds 4]

和 P0 同一套做法（模式 A：只給整張角色表，workflows/p0_ref_gen.json），但用 Lightning 4 步（P3 已驗證品質和 35 步相近）。
每個表情出多個 seed 到 refs/experiment/G0_expr/，挑好的再複製成 refs/04、05；Qwen 常忽略構圖，所以挑完可再裁成胸口以上。
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_keyframe_test as K

ROOT = K.ROOT
FRAMING = ("Close-up portrait from the chest up: only her head, shoulders and upper chest are in the image, the top of her head near the top edge, "
           "cut off at mid-chest; no arms crossed, no hands in the image. ")
COMMON = ("Single character only, no text, no letters, no captions, no symbols, no manga effect marks, no panel frames, plain pure white background, "
          "clean anime character-sheet illustration style identical to the reference. Keep the exact same character as the reference: same face with almond-shaped eyes "
          "and dark purple-grey irises, dark purple high ponytail with braids on the crown and a silver chain, small silver earrings, "
          "white sleeveless high-collar zip-up top. ")
TARGETS = [
    ("04_expr_happy", "Image 1 is a character reference sheet of the character. " + FRAMING + "Happy open-mouth smile, bright cheerful expression, eyes open and looking straight at the viewer. "),
    ("05_expr_angry", "Image 1 is a character reference sheet of the character. " + FRAMING + "Angry expression, furrowed eyebrows, eyes open and glaring straight at the viewer, mouth slightly open. "),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    out = ROOT / "refs/experiment/G0_expr"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "results.json"
    recs = json.loads(log.read_text()) if log.exists() else []
    for name, text in TARGETS:
        for k in range(a.seeds):
            dest = out / f"{name}_s{k}.png"
            if dest.exists():
                continue
            seed = 20261124 + 10 * k + int(name[:2])
            prompt = text + COMMON
            t0 = time.time()
            K.W, K.H = 832, 1024
            data = K.qwen(prompt, [], seed)
            dest.write_bytes(data)
            recs.append(dict(name=name, file=str(dest.relative_to(ROOT)), seed=seed, steps=4, lightning=True, size="832x1024", prompt=prompt,
                             seconds=round(time.time() - t0, 1)))
            log.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
            print(name, k, recs[-1]["seconds"], flush=True)


if __name__ == "__main__":
    main()
