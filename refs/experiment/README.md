# P0 參考圖實驗（Qwen-Image-Edit-2511 Q8_0）

> 本文件的「初步觀察」只是執行者的第一輪目視記錄；**最終外觀五項驗收由專案負責人（你）決定**。

## 設定
- 模型：`qwen-image-edit-2511-Q8_0.gguf`（來源 `unsloth/Qwen-Image-Edit-2511-GGUF`，21,761,817,184 bytes）
- sha256：`ab4f0622fb002fccaaa679a2ecce6fd1b3190d8ea28a5b7b2b17b8669bc24afa`（與 HF LFS oid 一致）
- 文字編碼器 `qwen_2.5_vl_7b_fp8_scaled`、VAE `qwen_image_vae`；**不使用 Lightning LoRA**
- Workflow：`workflows/p0_ref_gen.json`（API 格式，含兩張參考圖的版本；模式 A 由 `tools/p0_run_experiment.py` 移除 image2）。節點：UnetLoaderGGUF → ModelSamplingAuraFlow(shift 3) → CFGNorm → KSampler；正負向皆用 TextEncodeQwenImageEditPlus。
- 取樣：35 步、cfg 4.0、euler/simple；同一目標在 A、B 使用相同 seed 與 prompt（B 多一句描述第二張圖）。
- 輸出尺寸：全身 704x1024，表情與動作 832x1024（長邊約 1024）。
- 第二參考圖：`refs/crops.json` 與 `tools/crop_refs.py` 產生的 `refs/crops/*.png`。背面沒有對應格，B 模式改用側面格。
- 對照圖：`refs/experiment/compare.png`

## 每張參數與耗時（含模型載入與參考圖編碼）
| 模式 | 目標 | seed | steps | cfg | sampler | 尺寸 | 秒 |
|---|---|---|---|---|---|---|---|
| A | 01 full_front | 20251101 | 35 | 4.0 | euler/simple | 704x1024 | 339.2 |
| B | 01 full_front | 20251101 | 35 | 4.0 | euler/simple | 704x1024 | 546.2 |
| A | 02 side | 20251102 | 35 | 4.0 | euler/simple | 704x1024 | 327.1 |
| B | 02 side | 20251102 | 35 | 4.0 | euler/simple | 704x1024 | 528.2 |
| A | 03 back | 20251103 | 35 | 4.0 | euler/simple | 704x1024 | 348.1 |
| B | 03 back | 20251103 | 35 | 4.0 | euler/simple | 704x1024 | 528.2 |
| A | 04 expr_happy | 20251104 | 35 | 4.0 | euler/simple | 832x1024 | 345.3 |
| B | 04 expr_happy | 20251104 | 35 | 4.0 | euler/simple | 832x1024 | 570.2 |
| A | 05 expr_angry | 20251105 | 35 | 4.0 | euler/simple | 832x1024 | 360.1 |
| B | 05 expr_angry | 20251105 | 35 | 4.0 | euler/simple | 832x1024 | 555.2 |
| A | 06 action_point | 20251106 | 35 | 4.0 | euler/simple | 832x1024 | 375.2 |
| B | 06 action_point | 20251106 | 35 | 4.0 | euler/simple | 832x1024 | 555.2 |

平均：A 約 349 秒，B 約 547 秒（多一張參考圖，序列變長）。12 張合計約 90 分鐘。

## 峰值記憶體（每 2 秒取樣）
- RAM 已用 19129 MB、Swap 已用 10894 MB（實驗前 swap 已約 7–10 GB）、VRAM 15439 MB（16 GB 卡，幾乎滿載但未 OOM）。

## 初步觀察（對照外觀五項：臉、紫色編髮＋銀鏈、白色高領拉鍊上衣、黑色腿環在角色右腿（正面時在畫面左側、背面時在畫面右側）、手指數）
共通：紫色編髮與銀鏈、白色高領拉鍊上衣、腿環大致都在；手指只做縮圖等級目視，未逐張放大檢查。腿環位置已確認：在角色右腿，正面時在畫面左側、背面時在畫面右側。03 背面 A 版的腿環在畫面右側，符合。
- 01 全身正面：A 乾淨、單一角色、白底、無文字，表情偏平淡；B 把手寫台詞與星星抄進畫面（第二參考圖含文字），不合格。
- 02 側面：A、B 都是側面全身，無文字；腿環在側面可見，位置差異需放大確認。
- 03 背面：A 白底乾淨，馬尾與鏈子在背後，合理；B 背景變米色（不是白底），其餘相近。
- 04 開心：兩者都沒有照「頭肩特寫」，A 變成閉眼笑的腰上構圖，B 變成俯角全身；構圖不穩。
- 05 生氣：A 腰上構圖，抱胸，表情到位；B 俯角全身，背景偏灰白。
- 06 動作（指向）：A 指向側為畫面左，叉腰；B 指向鏡頭，較接近原格。兩者手指數看起來正常。
- 整體：A 背景較穩定地維持純白、沒有文字；B 的第二參考圖有時帶來文字、米色背景與非預期視角，但動作姿勢較貼近原格。表情的「頭肩特寫」需要改 prompt 或改用只含單格且去字的裁切。這些皆為初步印象。

## 其他
- 純文字歌詞檔尚未存在，已略過。
- `audio/song.wav` 長度 134.30 秒。
- 工具：`tools/crop_refs.py`、`tools/p0_run_experiment.py`、`tools/p0_compare.py`；`refs/experiment/results.json` 含完整 prompt。

## 決定（2026-10-15）
- 採用 A（只給整張角色表）。6 張 A 版複製到 `refs/` 作為正式參考圖：`01_full_front`、`02_side`、`03_back`、`04_expr_happy`、`05_expr_angry`、`06_action_point`。
- 之後用 Qwen-Image-Edit-2511 產生參考圖時，預設只給整張角色表，不加裁切單格。
