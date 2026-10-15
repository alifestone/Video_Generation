# 影片生成流程（Pipeline）

從角色表、歌曲與歌詞到 1920×1080、24fps 的完成片。整個流程分成 P0–P9 十個階段，中間有 G0–G4 五道審查。原則是先用 8 秒端到端試作（P4）決定模型與參數，通過後才依段落量產。

本檔只留流程本身。現況盤點、前置條件、環境建置、工時、風險、授權與參考資料見《LUNA MV 生成流程計畫書.md》。文中的「我」指專案負責人，也就是審查者。

## 原則

- 不做對嘴；品質優先，不以時間為限制。量化預設用 Q8 GGUF，放不下或 RAM 不夠時才降級。
- 一次只常駐一個大模型，依段落批次處理。
- 一個主 I2V 模型（由試作決定），加上少數舞蹈英雄鏡頭與視差鏡頭，不混用多個模型。
- 切點一律用 timeline 的絕對幀號，不用「各鏡時長累加」。
- 字標與歌詞不讓模型畫，後製時再疊加。
- 候選結果不進 git，只 commit 選定的版本。
- 由 Claude 執行的工作，沒有特別註明模型時一律使用 Claude Sonnet 5.5；目前只有 P2 劇本與 P5 鏡頭清單指定用 Claude Opus 5.5。

## 流程總覽

| 階段 | 內容 | 輸入 | 主要輸出 | 審查 |
| --- | --- | --- | --- | --- |
| P0 | 素材整理 | 角色表、`歌曲.mp3`、`歌詞.srt` | `refs/` 參考圖、`audio/song.wav` | 外觀五項驗收 |
| P1 | 音樂分析 | `audio/song.wav`、`audio/lyrics.srt` | `analysis/timeline.json` | 自動檢查＋聽感確認 |
| P2 | 概念與劇本 | timeline、歌詞、參考圖 | `docs/mv_plan.md`、`docs/style_bible.md` | 我確認段落規劃 |
| P3 | 角色資產與 LoRA | 參考圖、style bible、mv_plan | `luna_illustrious.safetensors`、風格樣張 | **G0：樣張與 LoRA** |
| P4 | 8 秒端到端試作 | LoRA、樣張、timeline、三個候選模型 | 3–4 鏡試作片、`docs/pilot_report.md` | 決定主模型、路線、是否訓練 Wan LoRA |
| P5 | 鏡頭清單與 animatic | mv_plan、timeline、pilot_report | `shotlist.json`、`edit/animatic.mp4` | **G1：animatic** |
| P6 | 關鍵幀（依段落） | shotlist、LoRA、style bible | `shots/sNNN/keyframe.png` | **G2：關鍵幀換進 animatic** |
| P7 | 影片生成（依段落） | 關鍵幀（與尾幀）、shotlist | `shots/sNNN/video.mp4` | **G3：粗剪** |
| P8 | 局部修補 | 有瑕疵的影片、SAM 遮罩 | 修補後的 `video.mp4` | 併入 G3 或 G4 |
| P9 | 後製 | 全部選定影片、原曲、字標 | `final/final.mp4` | **G4：定剪** |

P6 到 P8 依段落重複，建議從 Chorus 1 開始：一段做完、審過再做下一段，不要等全部關鍵幀出完才開始生影片。

## 各階段步驟

### P0 素材整理

1. 用 Qwen-Image-Edit-2511（Q8）以整張角色表為參考，一次產生一張獨立參考圖（例如「全身正面站姿，白背景，沒有文字」）。
2. 先做小實驗：出 6 張（全身正面、側面、背面、2 種表情、1 個動作），比較「只給整張角色表」和「整張角色表加裁出的單格」兩種輸入。
3. 驗收：每張只有一個角色、沒有文字；**外觀五項**（臉、紫色編髮與銀鏈、白色高領拉鍊上衣、左大腿黑色腿環、手指數）都正確。不通過就修改 prompt，或改用備案。
   - 備案 1：FLUX.2 Klein 9B 加 CharacterSheet LoRA；用 SAM 3 切出每個角色。
   - 備案 2：用 Pillow 腳本裁切（`tools/crop_refs.py`，座標寫在 `refs/crops.json`），再去字。
4. 音檔：`ffmpeg -i 歌曲.mp3 -c:a pcm_s16le audio/song.wav`，原始 mp3 一起保留。

### P1 音樂分析

1. allin1 在獨立的 CPU 環境執行，輸出拍點、首拍與段落；librosa 計算能量曲線（RMS 與 onset strength）。
2. 用 `tools/build_timeline.py` 合併 allin1、librosa 與 SRT，產出 `analysis/timeline.json`：每個拍點存絕對時間與 24fps 幀號，段落、能量、歌詞行都對應到小節編號。
3. 驗收：確認 BPM 是 108 還是 216；首拍聽起來正確；段落邊界與聽感的誤差小於 1 拍。

### P2 概念與劇本

由 Claude Opus 5.5 加 agency-agents Visual Storyteller 撰寫。

- `docs/mv_plan.md`：視覺概念、每段的情緒與敘事、每段的剪輯密度（例如 Intro 每 2 小節一刀、Chorus 每小節一刀），以及 3–6 個英雄鏡頭。
- `docs/style_bible.md`：外觀五項檢查清單、色票、光影、場景、禁止事項、每段的色調方向。
- 驗收：每句歌詞都有對應的畫面概念；由我確認段落規劃。

### P3 角色資產與 LoRA

1. 用 Qwen-Image-Edit-2511 從參考圖擴充 40–80 張不同角度、景別、光線、背景的圖，服裝保持一致。
2. 人工挑圖：外觀五項有任何一項錯就淘汰。
3. 用 kohya sd-scripts 訓練 Illustrious LoRA v1。
4. 用 v1 出 30 張，挑最好的 20–30 張加回資料集，訓練 v2。
5. 用正式管線（Illustrious 加 LoRA）出 6–10 張風格樣張，每個主要段落 1–2 張。
6. **G0**：樣張中的臉、髮型、服裝一致，色調與光影符合 style bible；確認後刪除其他圖片。

H3 產生的任何畫面都不能放進 LoRA 資料集，因為授權禁止用它的輸出訓練其他模型。

### P4 8 秒端到端試作

1. 依 mv_plan 與 timeline 選出代表性的點：舞蹈、特寫、視差字卡各 1 鏡，再加 1 鏡一般 I2V，共 3–4 鏡，從關鍵幀一路做到後製完成。
2. 用正式管線出關鍵幀（需要首尾幀的模型另出尾幀）。三個候選模型用相同的關鍵幀、動作描述與 seed，輸出放 `pilot/<模型>/shotNN.mp4`：
   - **Wan2.2 I2V A14B**：Q8 GGUF 的 high-noise 與 low-noise 兩個 expert，lightx2v 4-step 和完整步數各測一次。
   - **Index-AniSora V3.2**：動漫專用的 I2V，重點看線條與色塊在動作中會不會糊掉。
   - **MiniMax H3 FL2VA**：輸入首尾幀，模型與文字編碼器依序載入；turbo 8-step 和完整步數各測一次。
3. 每條 pipeline 都記錄生成時間、RAM 峰值、角色與畫風有沒有漂移、動作是否自然、首尾幀銜接。
4. 要回答的問題：
   - 主 I2V 模型用哪一個？
   - Q8 跑得動嗎？
   - 加速 LoRA 能不能當正式設定？
   - 舞蹈鏡頭用 Wan-Dancer 還是 Animate？
   - 手動關鍵姿勢加 Fun Control 是否可行？
   - 需不需要訓練 Wan LoRA？
   - RIFE 補幀與 Real-ESRGAN 放大會不會出現鬼影或閃爍？
5. 結果寫進 `docs/pilot_report.md`，並刪除未選定模型的輸出。**試作沒通過就不進入量產。**

### P5 鏡頭清單與 animatic

1. Claude Opus 5.5 依 mv_plan 與 timeline 產出 `shotlist.json`，內容包含每鏡的起訖幀、景別、運鏡、route、prompt 與等級（hero 或 normal）。prompt 依 Illustrious 的 Danbooru tag 與影片模型的格式撰寫。
2. animatic：每鏡先放一張草圖或風格樣張，在 Kdenlive 依時間軸配原曲排好，輸出 `edit/animatic.mp4`。
3. **G1**：Claude Sonnet 5.5 審查節奏與剪輯密度（切點是否落在拍點、密度是否符合 mv_plan）；段落起伏與其他意見由我審查。

### P6 關鍵幀（依段落）

1. 用 Illustrious 加 LUNA LoRA 出圖；Qwen-Image-Edit 只做局部修改、合成與換背景。運鏡幅度大的鏡頭同時出尾幀。
2. 每鏡出 4–8 張候選放在 `shots/sNNN/candidates/`，用 `ffmpeg xstack` 拼成對照圖，由我挑一張複製成 `keyframe.png`。
3. **G2**：把關鍵幀換進 animatic 一起看。

### P7 影片生成（依段落）

1. 依「影片路線」生成，每鏡頭尾各多生約 0.3–0.5 秒，剪輯時再裁掉。
2. 每鏡最多試 4 個 seed；都失敗就改路線或改設計（特寫、縮短、改成視差）。
3. 候選放在 `candidates/`，由我選 seed，Claude Sonnet 5.5 把選定的結果複製成 `video.mp4`。
4. **G3**：把選定影片換進 animatic，當成粗剪來看。

### P8 局部修補

用 Wan2.2-VACE-Fun-A14B 加 SAM 遮罩，修手指、銀鏈、腿環等局部瑕疵。驗收標準是瑕疵消失、周圍畫面不變。修好的檔案覆蓋同檔名後再 commit。

### P9 後製

1. **剪輯**（Kdenlive）：切點對齊 timeline 幀號；動作重音對不到拍點時，用 0.9–1.1 倍變速。
2. **幀率統一**：Wan 的 16fps 先用 RIFE 補到 48fps，再隔幀抽取成 24fps；H3 原生就是 24fps。
3. **放大**：1280×720 放大到 1920×1080，方法依試作結果決定。
4. **調色**（DaVinci Resolve）：定剪後轉成 DNxHR 匯入 Resolve，依 style bible 套統一的 LUT，再依段落微調。
5. **拍點特效**：閃白、震動、色差、速度變化、轉場，對準重拍。
6. **疊加**：CSFCCA 與 LIVE AI 字標用官方素材，歌詞字卡用字型排版。
7. **音訊**：一律用原曲。
8. 輸出 1920×1080、24fps 的 `final/final.mp4`。
9. **G4**：全片沒有跳幀、切點對拍、外觀五項全數通過。

## 影片路線

| 路線 | 適用 | 工具 |
| --- | --- | --- |
| A 一般 I2V（多數鏡頭） | 上半身律動、甩髮、轉頭、推拉搖移、特寫的眨眼與呼吸 | 主模型＋首幀（必要時加尾幀）＋文字 prompt |
| B 指定動作 | 副歌舞蹈英雄鏡頭（3–6 鏡）；比 V、指向鏡頭、轉身等短而精確的姿勢 | 舞蹈：Wan-Dancer-14B，備案 Wan2.2-Animate-14B 加替身驅動影片。姿勢：OpenPose 編輯器擺關鍵姿勢，加姿勢插值，經 Wan 2.2 Fun Control 生成 |
| C 靜態插畫 | 歌詞字卡、背景、插入鏡頭、不超過 1 小節的快切 | 關鍵幀＋Depthflow＋Depth Anything V2 或 3；位移要小，特寫不走視差 |

Bailando 與 HY-Motion 只在 B 路線的選項都失敗時才啟用。

## 時間與幀數規則

| 規則 | 內容 |
| --- | --- |
| 輸出規格 | 1920×1080、24fps |
| 剪輯時間軸 | 24fps，全曲 3223 幀 |
| 切點 | `round(拍點秒數 × 24)`，從 `timeline.json` 讀取 |
| Wan 生成長度 | 16fps、4n+1 幀：1 小節的鏡頭生 49 幀（3.06 秒），2 小節的鏡頭生 81 幀（5.06 秒） |
| H3 生成長度 | 24fps，依鏡頭長度加上頭尾餘量 |
| 鏡頭長度上限 | 一般不超過 2 小節（約 4.4 秒），避免後半段角色漂移 |
| 驗收 | 在剪輯時間軸上檢查「總長＝音檔長度」，不要求生成長度剛好等於鏡頭長度 |

## 審查與退回

不通過時，只退回到「能修好它的最後一步」。在 `review.csv` 填上代碼和一句說明，一個問題一列，只追加、不修改。

| 代碼 | 問題 | 退回到 | 重跑範圍 |
| --- | --- | --- | --- |
| S | 敘事或段落概念 | P2 | 該段落的劇本、鏡頭、關鍵幀、影片 |
| T | 節奏或切點 | P5 或 P9 | 先試重剪，不夠再重生該鏡頭 |
| K | 構圖、場景、色調 | P6（色調可能只要在 P9 調色） | 該鏡頭的關鍵幀與影片 |
| C | 角色走樣 | P6；多鏡走樣時回 P3 | 重訓 LoRA 時，未完成段落的關鍵幀全部重出 |
| M | 動作或穩定性 | P7 | 換 seed（上限 4 個）→ 改路線 → 改設計 |
| L | 局部瑕疵 | P8 | 只修該區域 |
| P | 畫質或流暢度 | P9 | 放大或補幀參數 |

```csv
round,reviewed_commit,gate,shot_id,artifact,artifact_sha256,verdict,code,t_in_shot,note,action,fixed_commit
r1,a1b2c3d,G3,s027,video.mp4,…,fail,M,1.8,右手變形,換 seed,
```

## 產出物與 commit

- 選定的檔名固定：`shots/sNNN/keyframe.png`、`pose.mp4`、`video.mp4`；鏡頭編號補零到 3 位。候選放在 `shots/sNNN/candidates/`，不進 git。
- 一批一個 commit，訊息寫段落、步驟與原因，例如 `chorus1 video: 選定 s021–s034；s027 M 換 seed`。
- 每道審查打一個 tag，例如 `g2-chorus1-r1`。
- 每次生成都記錄模型與 LoRA 的 sha256、workflow、seed、sampler、steps、cfg、shift、解析度、幀數，以及輸入檔的 sha256，記在 `shotlist.json`。目標是可追溯、近似重現。
