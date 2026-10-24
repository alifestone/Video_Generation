# lora/：LUNA 的 Illustrious LoRA（P3）

觸發詞 `lunacsf`。底模 `Illustrious-XL-v1.0`；訓練用 kohya sd-scripts（`~/creative/sd-scripts`，commit 11d8ced，torch 2.11.0+cu130）。
圖片與權重不進 git（見 `.gitignore`），可由下列檔案重建。

| 檔案 | 內容 |
| --- | --- |
| `../tools/p3_dataset_gen.py` | 候選圖生成（Qwen-Image-Edit-2511 Q8，只給整張角色表；60 個規格 × seed 0／1000） |
| `dataset_raw/results.json` | 每張候選圖的 prompt、seed、尺寸、耗時（不進 git 的資料夾裡；檔名可由規格重建） |
| `selection_v1.json` | 挑圖結果：保留清單（含實際構圖標籤、裁切）與淘汰清單（逐張原因） |
| `../tools/p3_prepare_dataset.py` | 依 selection 產生 `v1/dataset/<repeats>_lunacsf/`（圖＋Danbooru 標籤） |
| `train.sh`、`sample_prompts.txt` | 訓練指令與每 2 epoch 的取樣 prompt |

## v1（2026-10-24，Claude 自行執行，使用者授權）

- **候選圖**：60 規格 × 2 seed＝120 張。用 Lightning 4 步 LoRA（每張約 30–40 秒）；同 seed 對照 35 步（約 310 秒）品質相近，只少一點腰鏈細節，所以選 Lightning。
- **已知問題**：Qwen-Image-Edit 對「腰以上、只拍腿」這類構圖指令常常忽略，出成全身。處理方式是標籤照實際構圖寫，並從 4 張全身圖裁出腰以上。
- **挑圖**：照 style bible 外觀五項逐張目視，保留 93 張、淘汰 27 張，另加 4 張裁切，共 97 張。
  - 淘汰原因以腿環位置錯最多：背面把腿環畫在畫面左、側面朝畫面左卻畫在靠鏡頭的腿、出現兩條腿環。
  - 其他淘汰原因：招牌或螢幕上有類似文字的形狀、服裝走樣（白色過膝襪）、漫畫符號、兩張臉重疊的壞圖。
- **標籤**：只寫會變動的東西（構圖、角度、姿勢、表情、光線、背景）；固定外觀不寫，讓觸發詞吸收。
- **參數**：network_dim 32、alpha 16、只訓練 UNet、AdamW8bit、lr 1e-4（cosine，warmup 100）、10 epoch、repeats 4、batch 2、bf16、1024 bucket。
- **G0 前要請使用者複核**：挑圖由 Claude 代做；refs/04、05（表情參考圖）要不要回 P0 重出，留在 G0 決定（P0 的決定）。
- **v1 結果**：
  - 臉、紫色馬尾與銀鏈、白色上衣都穩定。
  - prompt 只寫觸發詞時，正面會被底模拉成黑色裙子＋黑色過膝襪；只要 prompt 寫明服裝標籤就正確。
  - 常見錯誤是腿環在錯的腿或兩腿都有，以及「特寫」prompt 被服裝標籤拉成膝上構圖。

## v2（2026-10-24）

- **資料集**：v1 的 97 張，加上 v1 評估圖（`v1/eval`，prompt 含服裝）中外觀五項全對的 18 張，共 115 張（`selection_v2.json`，另記 12 張淘汰原因）。
- **標籤**：改成依構圖寫出看得到的固定服裝（`p3_prepare_dataset.py --outfit visible`）：臉部特寫只寫頭髮與高領，全身才寫到鞋襪；固定寫 `single thigh strap`。
- **參數**：同 v1。拿掉 `shuffle_caption`，因為和快取文字編碼器輸出不能並用。
- **正式管線的 prompt**：一律帶服裝標籤（見 `tools/p3_lora_gen.py` 的 OUTFIT），負面加 `two thigh straps, thighhighs, kneehighs`。
