# LUNA MV 影片生成流程計畫書

Oct 14, 2026 · @周聖詠

先用 8 秒端到端試作決定模型與參數，通過後再依段落量產。本版整合硬體環境、工具查證、流程設計、git 與文件一致性 4 份評估，取代前一版計畫書。

## 和前一版的主要差異

最大的改變是「先試作、再依段落量產」，並且只用一個主影片模型。

| 項目 | 前一版 | 這一版 | 理由 |
| --- | --- | --- | --- |
| 試作 | 第 8 步才第一次看到動態畫面 | LoRA 完成後立刻做 8 秒端到端試作，用試作決定主模型 | RAM、變形、補幀鬼影要在量產前發現 |
| 審查 | 2 道 | 5 道（樣張 → animatic → 關鍵幀 → 粗剪 → 定剪），品質由你在各道審查評判 | 關鍵幀要放進節奏裡看 |
| 批次方式 | 先出完全部關鍵幀，再整批生影片 | 依段落：一段做完、審過再做下一段 | 問題在第一段就會暴露 |
| 影片模型 | H3、Wan Fun Control、視差三條路線並行 | 一個主 I2V 模型（試作決定）＋少數舞蹈英雄鏡頭＋視差 | 模型混太多，筆觸和色調會跳 |
| 舞蹈與動作 | Bailando、HY-Motion、影片替身、手動關鍵姿勢四種來源 | 舞蹈先試 Wan-Dancer-14B，備案 Wan2.2-Animate；手動關鍵姿勢保留，限短的精確姿勢；Bailando、HY-Motion 降為備案 | 少掉骨架比例重定向的變形風險；手動骨架可直接照 LUNA 比例擺 |
| 幀率 | 骨架重取樣成 24fps | Wan 以原生 16fps 生成，剪輯時間軸 24fps、以幀計算 | Wan 2.2 是 16fps、幀數 4n+1 |
| 參考圖 | Claude Sonnet 5.5 透過 GIMP MCP 裁切角色表 | Qwen-Image-Edit-2511 以整張角色表為輸入，生成獨立參考圖；不行再退回裁切加去字 | 手寫字與漫畫符號和角色重疊，矩形裁切去不掉；整張輸入可保留畫風資訊 |
| LoRA 資料 | 只有角色表裁圖 | 參考圖 → Qwen-Image-Edit 擴充 40–80 張 → 人工挑選 | 一張角色表的素材太少 |
| 關鍵幀與樣張 | Qwen-Image-Edit 出樣張與關鍵幀 | Illustrious 加 LoRA 為主，Qwen-Image-Edit 只做局部修改；樣張改在 LoRA 之後用正式管線出 | 樣張必須能用正式管線重現 |
| agency-agents | Visual Storyteller、Brand Guardian、Image Prompt Engineer | 只留 Visual Storyteller | Brand Guardian 偏商業品牌策略；Image Prompt Engineer 專攻攝影風格 prompt，不熟 Illustrious 的 Danbooru tag 和影片 prompt（查證） |
| 後製軟體 | Kdenlive 或 DaVinci Resolve | Kdenlive 剪輯、DaVinci Resolve 調色 | Resolve 調色較強；Linux 免費版不能解 H.264，定剪後轉 DNxHR 再進 Resolve |
| 版本控管 | 每個鏡頭重生一次就 commit 一次；tag `review-r1` | 候選放在不進 git 的 `candidates/`，選定的才 commit；一批一個 commit；tag 依審查與段落命名 | LFS 空間改寫歷史也不能釋放（查證） |
| ComfyUI | 在 `~/mv/ComfyUI` 新裝，PyTorch cu128 | 沿用現有 `~/creative/ComfyUI`（已是 torch 2.11.0+cu130），只更新與補裝節點，不用舊 workflow | 省空間；現有環境已符合 Blackwell 需求（本機） |
| 啟動參數 | `--cache-none` 或 `--disable-smart-memory` | `--fast-disk --disable-pinned-memory --reserve-vram 1.5` | `--disable-smart-memory` 反而增加 RAM 壓力（本機原始碼） |
| Swap | 開 32GB swapfile | zram 加現有 16G swapfile、earlyoom | 磁碟只剩 75G；zram 不佔磁碟、比 swapfile 快 |
| allin1 | 照 NATTEN 官網安裝 | 另開純 CPU 舊版環境 | 照官網裝一定會失敗（查證） |
| 刪掉的 | — | PenShot；Brand Guardian、Image Prompt Engineer；GIMP MCP；ComfyUI-HyMotion 節點與 Bailando 環境的安裝步驟（備案需要時再裝） | 不合用，或隨工具降為備案 |

## 前提與原則

- **不需要對嘴**：LUNA 不唱歌，所有 lip-sync 工具都排除在外，包括 Wan2.2-S2V。
- **非商業用途，影片只給自己看**：可以使用非商用授權的模型。
- **不以時間為限制**：品質優先，可以選慢但好的設定，量化預設 Q8 GGUF，放不下或 RAM 不夠時才降級。
- **不沿用任何先前的記憶、先前的生成結果或先前的工作流程設定**。沿用的只有現有的 ComfyUI 安裝與模型權重檔。
- **品質由你評判**：透過 5 道審查與退回代碼決定是否通過。
- **一次只常駐一個大模型**；依段落批次處理。
- **版本追蹤用 git**：遠端倉庫 `git@github.com:alifestone/Video_Generation.git`，檔名不帶版本號。
- **Claude 模型預設 Sonnet 5.5**：由 Claude 執行的工作，沒有特別註明模型時一律使用 Claude Sonnet 5.5；目前只有 P2 劇本與 P5 鏡頭清單指定用 Claude Opus 5.5。

| 標記 | 意思 |
| --- | --- |
| 查證 | 2026 年 10 月上網查到的資料 |
| 本機 | 2026-10-14／15 在這台機器上實際檢查過 |
| 建議 | 規劃判斷 |
| 需實測 | 要實際跑過才知道 |

## 現況盤點

硬體夠用，但磁碟只剩 75G、swap 平常就用掉 10.7G，這兩項要先處理。

### 硬體與系統（本機）

| 項目 | 規格 | 對流程的影響 |
| --- | --- | --- |
| CPU | Ryzen 7 7700（8C/16T） | 音樂分析、ffmpeg、補幀的 CPU 部分夠用 |
| GPU | RTX 5060 Ti 16GB（Blackwell，sm\_120），驅動 580.178（支援 CUDA 13.0） | Hyprland 加瀏覽器約佔 1GB VRAM，要預留 |
| RAM | 32GB DDR5 | offload 的上限。H3 的擴散模型和文字編碼器加起來超過 RAM，必須依序載入 |
| Swap | 16G swapfile，目前已用 10.7G | 開工前要查出是誰佔用 |
| 儲存 | Linux `/` 分割區 490G，只剩 75G 可用；另約 430G 是 Windows 分割區；外接 2TB 硬碟（創見 StoreJet 25M3） | 見「前置條件」 |
| OS | Arch Linux（Hyprland／Wayland） | 每個工具用 uv 建獨立 Python 環境 |

### 素材

| 素材 | 狀態 | 處理 |
| --- | --- | --- |
| `歌曲.mp3` | 134.3 秒、44.1kHz 立體聲、128kbps，內嵌「AI 生成內容」標記 | 保留原檔，另轉一份 `audio/song.wav`。allin1 官方說不同 MP3 解碼器會偏移 20–40 ms（查證） |
| `歌詞.srt` | 32 行，0:00.68–2:12.22，斷句已對齊人聲 | 與拍點、段落合併成時間軸 |
| 歌詞純文字（待放入 repo） | 與 SRT 內容大致相同，部分字詞不一（如「訊息／訊號」「指尖／之間」） | **以 SRT 為準** |
| `動畫_女生_LUNA.jpg` | 1920×1080 角色表：全身站姿、4 張動作、9 種表情、個性說明；上方有網站導覽列，左側有「LIVE AI LUNA」直寫字；全身圖旁有手寫台詞，多格裡有星星、問號、怒氣符號等漫畫符號（本機） | 見 P0 |

### 歌曲結構（librosa 初步分析，正式數字以 P1 為準）

約 108 BPM：每拍 0.556 秒、每小節 2.222 秒。24fps 下每拍 13.33 幀、每小節 53.33 幀，全曲 3223 幀。電子曲的節拍偵測常差一倍，也可能是 216 BPM，P1 要確認。

| 時間 | 能量（全曲最大＝10） | 歌詞（SRT 行號） | 段落 |
| --- | --- | --- | --- |
| 0:00–0:12 | 4 | 1–2「訊號亮起…世界重啟」 | Intro |
| 0:12–0:20 | 8 | 3「321 LIVE AI」 | Hook |
| 0:20–0:37 | 6 | 4–6 | Verse |
| 0:37–0:48 | 6–7 | 7–11 | Pre-chorus |
| 0:48–1:17 | 7–8 | 12–18 | Chorus 1 |
| 1:17–1:25 | 7–8 | 19–20（英文） | Bridge |
| 1:25–1:33 | 6–7 | 21 | Build（能量短暫下降） |
| 1:33–2:10 | 9–10 | 22–31 | Final chorus（全曲最高點） |
| 2:10–2:14 | 5→0 | 32「CSFCCA LIVE AI」 | Outro |

### 既有 ComfyUI 與模型權重（本機，`~/creative/ComfyUI`）

ComfyUI 0.34.0（commit e80c1570，2026-09-03），venv 為 Python 3.12、torch 2.11.0+cu130、triton 3.6.0；SageAttention 未安裝。

| 處置 | 檔案 | 說明 |
| --- | --- | --- |
| 沿用 | `Illustrious-XL-v1.0`（6.5G） | 關鍵幀主模型 |
| 沿用 | `qwen-image-edit-2511-Q4_K_M.gguf`（12.3G）、Lightning 4-step LoRA、`qwen_2.5_vl_7b_fp8_scaled`、`qwen_image_vae` | 預設改用 Q8（需下載，約 21.8G，超過 16GB VRAM，需 offload）；Q4\_K\_M 留作備援 |
| 沿用 | H3 全套：FL2VA pruned INT8（19.5G）、Qwen3-VL NVFP4 文字編碼器（14.6G）、video／audio VAE、turbo 8-step LoRA | 試作時參加主模型比較 |
| 沿用 | `umt5_xxl_fp8_e4m3fn_scaled`、`wan_2.1_vae`、`wan2.2_vae` | Wan 系列共用 |
| 沿用 | ControlNet promax、BiRefNet | 構圖控制、去背 |
| 移到外接碟 | `Wan2.2-S2V-14B-Q4_K_M`（12.9G）、`wav2vec2_large`（0.6G） | 對嘴用，本計畫用不到 |
| 移到外接碟 | `wan2.2_ti2v_5B_fp16`（9.3G）、`Wan2.2-TI2V-5B-Q8_0`（5.0G） | 5B 畫質不如 14B |
| 要下載（Q8 為主） | Qwen-Image-Edit-2511 Q8、Wan2.2 I2V A14B、lightx2v 加速 LoRA、Wan-Dancer-14B、Wan2.2-Animate-14B、Wan 2.2 Fun Control、Wan2.2-VACE-Fun-A14B、Index-AniSora V3.2、Depth Anything V2 或 3、RIFE、Real-ESRGAN anime | 總量需實測；Q8 比 Q6 大，磁碟會更緊 |

## 前置條件

開工前先完成 1–3；第 4 項最晚在試作結束、進入量產前完成。

1. **釋放磁碟空間（約 57G，可用從 75G 到約 130G）**：
   - 搬到外接碟（約 29G）：S2V 14B 與 wav2vec2、TI2V 5B 兩個版本、前幾次的 LUNA MV 結果（`futuremode_2026_hackathon/luna-mv`、`ComfyUI/output`；本計畫不使用，只封存）。
   - 直接清除快取（約 28G）：`uv cache prune`（最多約 17.6G）、`paccache -rk1`（約 5.2G）、`pip cache purge`（約 3.3G）、`journalctl --vacuum-size=200M`（約 2.2G）。
   - 選擇性：Steam 遊戲庫移到外接碟（約 15G）。
   - `~/research`、`~/miniconda3`、`~/models/bonsai` 近兩週有更動，是否搬移由你決定。
   - 外接 2TB 硬碟（USB 3.1 Gen 1 傳統硬碟，約 100–130 MB/s）只放備份、封存的候選、暫時不用的權重；不拿來載入模型、放 swap 或放工作目錄。格式化成 ext4。
2. **記憶體**：
   - 查出誰佔了 10.7G swap：`for f in /proc/*/status; do awk '/^Name|^VmSwap/{printf $2" "}END{print ""}' $f; done | sort -k2 -n | tail`。
   - 加 zram（zstd，8–16G，優先順序高於 swapfile）；現有 16G swapfile 保留。
   - 安裝 earlyoom 或啟用 systemd-oomd；生成時關掉瀏覽器或它的硬體加速。
3. **git repo 初始化**（目前沒有任何 commit，本機也還沒裝 git-lfs）：第一個 commit 只放 `.gitattributes` 和 `.gitignore`，之後才加素材。確認 GitHub 倉庫是 private。
4. **從 Windows 分割區切出空間**，讓 `/` 可用至少 250GB。預設 Q8 時，130G 能否撐過試作需實測；量產與加訓 Wan LoRA（另需約 57G）一定不夠。切割前先備份。

## 流程總覽

以下的"我"和"你"皆指"周聖詠"

流程分成 P0–P9 十個階段，中間有 G0–G4 五道審查；P4 試作是是否量產的關卡。

| 階段 | 內容 | 主要輸出 | 審查 |
| --- | --- | --- | --- |
| P0 | 素材整理 | `refs/`、`audio/song.wav` | — |
| P1 | 音樂分析 | `analysis/timeline.json` | 自動檢查＋聽感確認 |
| P2 | 概念與劇本 | `docs/mv_plan.md`（含段落劇本）、`docs/style_bible.md` | — |
| P3 | 角色資產與 LoRA | `lora/dataset/`、`luna_illustrious.safetensors`、風格樣張 | **G0：樣張與 LoRA** |
| P4 | 8 秒端到端試作 | 3–4 鏡試作片、`docs/pilot_report.md` | 決定主模型、路線、是否訓練 Wan LoRA |
| P5 | 鏡頭清單與 animatic | `shotlist.json`、`edit/animatic.mp4` | **G1：animatic** |
| P6 | 關鍵幀（依段落） | `shots/sNNN/keyframe.png` | **G2：關鍵幀換進 animatic 後審查** |
| P7 | 影片生成（依段落） | `shots/sNNN/video.mp4` | **G3：粗剪** |
| P8 | 局部修補 | 修好的片段 | 併入 G3 或 G4 |
| P9 | 後製 | `final/final.mp4` | **G4：定剪** |

P6 到 P8 依段落重複：第一個段落建議用 Chorus 1，做完、審過再做下一段，不要等全部關鍵幀出完才開始生影片。

## 各階段細節

每個階段都有輸出與驗收標準；試作（P4）沒通過就不進入量產。

### P0 素材整理

**輸入**：`動畫_女生_LUNA.jpg`（角色表）、`歌曲.mp3`、`歌詞.srt`、歌詞純文字。

**輸出**：`refs/` 獨立參考圖（含實驗用的 `crops.json`）、`audio/song.wav`；原始 mp3 與歌詞整理進 `audio/`。

- **參考圖用 Qwen-Image-Edit-2511 從整張角色表生成**（Q8 為主）：每次以整張角色表為參考，下一個指令產生一張獨立參考圖，例如「畫出這位角色的全身正面站姿，白背景，沒有文字」。官方說 2511 角色一致性大幅提升，新視角可直接由基礎模型生成，並支援多張參考圖輸入（查證）。
- **先做小實驗**：出 6 張（全身正面、側面、背面、2 種表情、1 個動作），比較「只給整張角色表」與「整張角色表加裁出的單格」兩種輸入。整張圖輸入時會被縮小，編髮與銀鏈的細節可能流失（需實測）。
- 驗收：每張只有一個角色、沒有文字；外觀五項（臉、紫色編髮與銀鏈、白色高領拉鍊上衣、黑色腿環在角色右腿（正面時在畫面左側、背面時在畫面右側）、手指數）正確，並由我（周聖詠）驗收。如果通過，繼續下一個階段。如果不通過，根據我的判斷修改 Qwen-Image-Edit-2511 的 prompt 或是使用下方備案
- 備案 1：FLUX.2 Klein 9B 加 CharacterSheet LoRA（輸出多視角角色表，以真人訓練為主）；SAM 3 自動切出每個角色。

* 備案 2：**用 Pillow 腳本裁切**（`tools/crop_refs.py`），座標寫在 `refs/crops.json`，可以重跑。用於實驗的單格參考，以及整張輸入不行時的退回方案（裁切加去字，去字方法屆時再決定）。
* 音檔：`ffmpeg -i 歌曲.mp3 -c:a pcm_s16le audio/song.wav`，原始 mp3 一起保留。

### P1 音樂分析

**輸入**：`audio/song.wav`、`audio/lyrics.srt`。

**輸出**：`analysis/timeline.json`（BPM、拍點與首拍的絕對時間與 24fps 幀號、段落、能量曲線、歌詞行）；allin1 原始輸出 `analysis/struct/`。

- allin1 在獨立的 CPU 環境執行，輸出拍點、首拍、段落；librosa 計算能量曲線（RMS 與 onset strength）。
- `tools/build_timeline.py` 合併 allin1、librosa 和 SRT，產出 `analysis/timeline.json`：每個拍點存絕對時間與 24fps 幀號，段落、能量、歌詞行對應到小節編號。
- 驗收：BPM 確認是 108 還是 216；首拍聽起來對；段落邊界和聽感誤差小於 1 拍。

### P2 概念與劇本

**輸入**：`analysis/timeline.json`、`audio/lyrics.srt`、`refs/` 參考圖。

**輸出**：`docs/mv_plan.md`（視覺概念、段落劇本、剪輯密度、英雄鏡頭）、`docs/style_bible.md`（外觀五項、色票、光影、場景、禁止事項、每段色調）。

由 Claude Opus 5.5 加 agency-agents Visual Storyteller 撰寫。

- `docs/mv_plan.md`：視覺概念、每段的情緒與敘事；每段的剪輯密度（例如 Intro 每 2 小節一刀、Chorus 每小節一刀）；3–6 個英雄鏡頭（建議）。
- `docs/style_bible.md`：LUNA 外觀檢查清單（五項）、色票、光影、場景、禁止事項、每段的色調方向。
- 驗收：每句歌詞都有對應的畫面概念；由我確認段落規劃。

### P3 角色資產與 LoRA

**輸入**：`refs/` 參考圖、`docs/style_bible.md`、`docs/mv_plan.md`（樣張要對應的段落）。

**輸出**：`lora/dataset/`（挑選後的訓練圖）、`lora/` 訓練設定、`luna_illustrious.safetensors`（v1、v2，放外接碟）、6–10 張風格樣張（同時用來檢查 LoRA 角色一致性）。

1. 用 Qwen-Image-Edit-2511 從 P0 的參考圖擴充 40–80 張不同角度、景別、光線、背景的圖，服裝保持一致。
2. 你挑圖：外觀五項任一項錯就淘汰。
3. 用 kohya sd-scripts 訓練 Illustrious LoRA v1。
4. 用 v1 出 30 張，挑最好的 20–30 張加回資料集，訓練 v2。
5. 用正式管線（Illustrious 加 LoRA）出 6–10 張風格樣張，每個主要段落 1–2 張。
6. **G0 審查**：風格樣張中 LUNA 的臉、髮型、服裝細節一致，且色調、光影符合 style bible；由我確認。
7. 確認後將其他圖片刪除。

H3 生成的任何畫面都不能放進 LoRA 資料集：H3 授權禁止用它的輸出改進其他 AI 模型（查證）。

### P4 8 秒端到端試作

**輸入**：LUNA LoRA 與風格樣張（P3）、`docs/mv_plan.md` 的 Hook 段規劃、`analysis/timeline.json`（0:12–0:20）、`audio/song.wav`（舞蹈鏡頭用）、三個候選影片模型。

**輸出**：`pilot/<模型>/shotNN.mp4`、後製完成的 Hook 段試作片、`docs/pilot_report.md`（主模型、Q8 可否、路線、是否訓練 Wan LoRA、實測時間與 RAM 峰值）。

試作片段不先預設。到 P4 時先看 docs/mv\_plan.md 和 timeline.json，找出哪裡有舞蹈、哪裡需要特寫、哪裡會用到視差字卡，各選一個代表性的點，再加一鏡一般 I2V，共 3–4 鏡，從關鍵幀一路做到後製完成。視差字卡：把字標或歌詞拆成前景、中景、背景等幾層，再用不同速度移動，做出有深度的文字畫面。

三個候選模型用同一批輸入、各跑一條測試 pipeline，結果才能直接比較：

- **共同輸入**：從 P3 的正式管線（Illustrious 加 LUNA LoRA）出關鍵幀，每鏡一張；同時出尾幀給需要首尾幀的模型。三個模型用相同的關鍵幀、動作描述和 seed，輸出放 `pilot/<模型>/shotNN.mp4`。
- **Wan2.2 I2V A14B**：在 ComfyUI 跑 Q8 GGUF 的 high-noise／low-noise 兩個 expert，各測 lightx2v 4-step 與完整步數；輸出約 5 秒、16fps。
- **Index-AniSora V3.2（動漫專用）**：同樣用關鍵幀做 I2V，依其官方倉庫或 ComfyUI 節點執行；重點看動漫線條與色塊會不會在動作中糊掉。
- **MiniMax H3 FL2VA**：在 ComfyUI 本機執行（依序載入模型與文字編碼器），輸入首尾幀；加 turbo 8-step 與完整步數比較。
- **每條 pipeline 都記錄**：生成時間、RAM 峰值（`nvtop`、`free -h`）、角色與畫風有沒有漂移、動作是否自然、首尾幀銜接；全部寫進 `docs/pilot_report.md`。

| 要回答的問題 | 比較項目 |
| --- | --- |
| 主 I2V 模型用哪個？ | Wan2.2 I2V A14B、Index-AniSora V3.2（動漫專用）、MiniMax H3 FL2VA |
| Q8 跑得動嗎？ | 每個模型的 RAM 峰值與生成時間；放不下或 RAM 不夠才降到 Q6 |
| 加速 LoRA 能不能當正式設定？ | Wan 加 lightx2v 4-step、H3 加 turbo 8-step，和完整步數比較 |
| 舞蹈鏡頭怎麼做？ | Wan-Dancer-14B vs Wan2.2-Animate（驅動影片用 Wan T2V 生的替身） |
| 手動關鍵姿勢可行嗎？ | 手擺骨架加姿勢插值，經 Fun Control 生成 |
| 需不需要 Wan LoRA？ | 看 5 秒內角色有沒有漂移 |
| 補幀與放大會不會出問題？ | RIFE 16→48→24fps 的鬼影；Real-ESRGAN 逐幀放大的閃爍 |
| Illustrious 加 LoRA 的關鍵幀餵進 I2V 後會不會跑掉？ | 三個模型各看 5 秒內角色臉、髮型、服裝與畫風有沒有漂移，和 P3 的樣張對照 |

結果寫成 `docs/pilot_report.md`，據此更新影片路線與工時估算。試作沒通過就不進入量產。

確認後，將未選定的兩個模型輸出刪除。

### P5 鏡頭清單與 animatic

**輸入**：`docs/mv_plan.md`、`analysis/timeline.json`、`docs/pilot_report.md`（路線與模型決定）、風格樣張、`audio/song.wav`。

**輸出**：`shotlist.json`（每鏡的起訖幀、景別、運鏡、route、prompt、等級）、`edit/animatic.mp4` 與 Kdenlive 專案。

- Claude Opus5.5 依 `mv_plan.md` 與 `timeline.json` 產出 `shotlist.json`，prompt 依 Illustrious（Danbooru tag）與影片模型的格式撰寫。
- 切點一律用 timeline 的絕對幀號，不用「各鏡時長累加」。
- animatic：每鏡先放一張草圖或風格樣張，用 Kdenlive 照時間軸配原曲排好，輸出 `edit/animatic.mp4`。
- **G1 審查**：節奏和剪輯密度由 Claude Sonnet 5.5 審查（切點是否落在拍點、各段密度是否符合 mv\_plan.md）；段落起伏與其他意見由你審查。

### P6 關鍵幀（依段落）

**輸入**：`shotlist.json`（該段落的鏡頭）、LUNA LoRA、`docs/style_bible.md`、`refs/` 參考圖。

**輸出**：`shots/sNNN/keyframe.png`（必要時加尾幀）、`shots/sNNN/candidates/` 候選圖（不進 git）、更新後的 animatic。

- 主模型是 Illustrious 加 LUNA LoRA；Qwen-Image-Edit 只做局部修改、合成、換背景。
- 大幅運鏡的鏡頭同時出尾幀。
- 每鏡出 4–8 張候選放在 `shots/sNNN/candidates/`，用 `ffmpeg xstack` 拼成對照圖，由我依自己的判斷挑一張（目前沒有固定規則）複製成 `keyframe.png`。
- 字標和歌詞不讓模型畫，P9 再疊加。
- **G2 審查**：把關鍵幀換進 animatic 一起看。

### P7 影片生成（依段落）

**輸入**：`shots/sNNN/keyframe.png`（與尾幀）、`shotlist.json` 的 route、prompt、幀數；路線 B 另加 `audio/song.wav` 片段（Wan-Dancer）、替身驅動影片（Animate）或手動骨架（Fun Control）。

**輸出**：`shots/sNNN/video.mp4`、路線 B 的 `shots/sNNN/pose.mp4`、`candidates/` 候選影片（不進 git）、更新後的粗剪。

- 依「影片路線與模型」生成，每鏡頭尾各多生約 0.3–0.5 秒，剪輯時裁切。
- 每鏡最多 4 個 seed；都失敗就改路線或改設計（特寫、縮短、改視差）。
- 候選放 `candidates/`。由我選擇每個鏡頭用哪個 seed 的產出。Claude sonnet 5.5 將選定的鏡頭複製成 `video.mp4`。
- **G3 審查**：把選定影片換進 animatic，變成粗剪來看。

### P8 局部修補

**輸入**：有局部瑕疵的 `shots/sNNN/video.mp4`、`review.csv` 裡 L 類問題的位置與說明、SAM 遮罩。

**輸出**：修補後的 `shots/sNNN/video.mp4`（覆蓋同檔名再 commit）。

- Wan2.2-VACE-Fun-A14B 加 SAM 遮罩，修手指、銀鏈、腿環等局部瑕疵。
- 驗收：瑕疵消失，周圍畫面不變。

### P9 後製

**輸入**：全部選定的 `shots/sNNN/video.mp4`、粗剪的 Kdenlive 專案、`analysis/timeline.json`、`audio/song.wav`（原曲）、`docs/style_bible.md`（色調）、CSFCCA／LIVE AI 官方字標素材、`audio/lyrics.srt`（歌詞字卡）。

**輸出**：`final/final.mp4`（1920×1080、24fps）、Kdenlive 專案、Resolve 專案匯出（`.drp`）。

1. **剪輯**（Kdenlive）：切點對齊 timeline 幀號；動作重音對不到拍點時，用 0.9–1.1 倍變速。
2. **幀率統一**：Wan 的 16fps 用 RIFE 補到 48fps 再隔幀抽取成 24fps；H3 原生 24fps。
3. **放大**：1280×720 → 1920×1080，方法依試作結果。
4. **調色**（DaVinci Resolve）：定剪後轉 DNxHR 匯入 Resolve，依 style bible 做統一 LUT，再依段落微調。Resolve 在 Hyprland 上的穩定性需實測。
5. **拍點特效**：閃白、震動、色差、速度變化、轉場，對準重拍。
6. **疊加**：CSFCCA／LIVE AI 字標用官方素材，歌詞字卡用字型排版。
7. **音訊**：一律用原曲。H3 的音訊輸出不接 `VAEDecodeAudio` 就是靜音。
8. 輸出 1920×1080、24fps。
9. **G4 審查**：全片無跳幀、切點對拍、外觀五項全數通過。

## 影片路線與模型

主模型由 P4 從三個候選中選一個；另外兩個最多留給特定鏡頭。

| 路線 | 適用 | 工具 | 備註 |
| --- | --- | --- | --- |
| A 一般 I2V（多數鏡頭） | 上半身律動、甩髮、轉頭、推拉搖移、特寫的眨眼與呼吸 | 主模型＋首幀（必要時加尾幀）＋文字 prompt | Wan 系列：16fps、81 幀（約 5 秒）、幀數 4n+1。H3：24fps、最長 15 秒、短邊 768px（查證） |
| B 指定動作 | 副歌舞蹈英雄鏡頭（3–6 鏡）；比 V、指向鏡頭、轉身等短的精確姿勢 | 舞蹈：首選 Wan-Dancer-14B（音樂加角色參考圖，查證），備案 Wan2.2-Animate-14B 加替身驅動影片。精確姿勢：ComfyUI OpenPose 編輯器擺關鍵姿勢、姿勢插值補中間幀，經 Wan 2.2 Fun Control 生成 | Dancer 一次採樣 149 幀、很吃 VRAM、沒測過動漫角色（需實測）。手動骨架直接照 LUNA 比例擺，鏡頭角度要和關鍵幀一致；插值效果需實測 |
| C 靜態插畫 | 歌詞字卡、背景、插入鏡頭、≤ 1 小節的快切鏡頭 | 關鍵幀＋Depthflow（本機已裝）＋Depth Anything V2 或 3 | 快切段內和路線 A 交錯，避免整段太靜；位移要小，動漫髮絲在深度邊緣容易被拉扯。特寫不走視差 |

Bailando 與 HY-Motion 只在 B 路線的選項都不行時才啟用。HY-Motion 官方最低 VRAM 24–26GB，要用 GGUF 版文字編碼器才放得進 16GB（查證）。

## 時間與幀數規則

輸出 1920×1080、24fps；剪輯時間軸全曲 3223 幀，切點取拍點的絕對幀號。

| 規則 | 內容 |
| --- | --- |
| 輸出規格 | 1920×1080、24fps |
| 剪輯時間軸 | 24fps，全曲 3223 幀 |
| 切點 | `round(拍點秒數 × 24)`，從 `timeline.json` 讀取 |
| Wan 生成長度 | 4n+1 幀、16fps。1 小節鏡頭生 49 幀（3.06 秒），2 小節鏡頭生 81 幀（5.06 秒） |
| H3 生成長度 | 24fps，依鏡頭長度加頭尾餘量 |
| 驗收 | 「總長＝音檔長度」在剪輯時間軸上驗收，不要求生成長度剛好等於鏡頭長度 |
| 鏡頭長度上限 | 一般 ≤ 2 小節（約 4.4 秒），避免後半段角色漂移 |
| 預估鏡頭數 | 約 45–55 鏡（建議） |

## 審查與退回

品質由你在五道審查中評判。不通過時，只退回到「能修好它的最後一步」；你只要在 `review.csv` 填代碼與一句說明。

| 代碼 | 問題類型 | 例子 | 退回到 | 重跑範圍 |
| --- | --- | --- | --- | --- |
| S | 敘事或段落概念 | 「副歌太平淡」 | P2 | 該段落的劇本、鏡頭、關鍵幀、影片 |
| T | 節奏或切點 | 「鏡頭太長」「沒對到鼓點」 | P5 或 P9 | 先試重剪，不夠再重生該鏡頭 |
| K | 構圖、場景、色調 | 「背景不對」「太暗」 | P6（色調可能只要 P9 調色） | 該鏡頭的關鍵幀與影片 |
| C | 角色走樣 | 「臉不像」「髮色變了」 | P6；多鏡走樣時回 P3 | 重訓 LoRA 時，未完成的段落全部重出關鍵幀 |
| M | 動作或穩定性 | 「動作歪掉」「畫面閃爍」 | P7 | 換 seed（上限 4 個）→ 改路線 → 改設計 |
| L | 局部瑕疵 | 「手指多一根」「腰帶鎖鏈糊掉」 | P8 | 只修該區域 |
| P | 畫質或流暢度 | 「太糊」「卡頓」 | P9 | 放大或補幀參數 |

`review.csv` 一個問題一列，只追加不修改；`verdict` 是 pass、fail 或 hold，`gate` 是 G0–G4。

```csv
round,reviewed_commit,gate,shot_id,artifact,artifact_sha256,verdict,code,t_in_shot,note,action,fixed_commit
r1,a1b2c3d,G3,s027,video.mp4,…,fail,M,1.8,右手變形,換 seed,
```

`shotlist.json` 另標記每鏡等級（hero 或 normal），算力優先給 hero。

## 版本控管與儲存

git 存文字，Git LFS 只存通過審查的媒體檔，候選與原始輸出不進 git、定期封存到外接碟。

- **git**：shotlist、timeline、workflow JSON、腳本、文件、review.csv。
- **Git LFS**：選定的 keyframe、pose、video，refs，audio，里程碑版的 final（只在 G4 通過時 commit）。
- **額度**：GitHub LFS 免費 10 GiB 儲存、10 GiB 流量；**改寫歷史不會釋放 LFS 空間，只能刪掉 repo 重建**（查證）。照上述原則預估用量 4–6 GiB（建議）。
- **備份**：外接碟上放一份 `git clone --mirror` 加 `git lfs fetch --all`。

### 目錄

```text
Video_Generation/
├─ .gitattributes / .gitignore
├─ refs/                 # 參考圖、crops.json
├─ audio/                # 歌曲.mp3（原檔）、song.wav、lyrics.srt、歌詞純文字
├─ analysis/timeline.json
├─ docs/                 # mv_plan、style_bible、pilot_report
├─ tools/                # crop_refs.py、build_timeline.py 等腳本
├─ workflows/            # ComfyUI workflow（API 格式 JSON）、custom node snapshot
├─ lora/                 # 訓練設定、資料集說明
├─ shotlist.json
├─ review.csv
├─ shots/s014/           # keyframe.png、pose.mp4、video.mp4（固定檔名）
│  └─ candidates/        # 不進 git
├─ edit/                 # Kdenlive 專案、animatic.mp4、Resolve 專案匯出（.drp）
└─ final/
```

鏡頭編號補零到 3 位（`s014`），排序才正確。

### `.gitattributes` 與 `.gitignore`

LFS 追蹤（每個 pattern 一行，用 `git lfs track` 產生）：`*.png *.jpg *.jpeg *.webp *.gif *.tif *.exr *.psd *.xcf *.mp4 *.mov *.mkv *.webm *.wav *.mp3 *.flac *.fbx *.blend *.bvh *.npy *.npz` \*.drp。

```text
# .gitignore
**/candidates/
.venv/
__pycache__/
cache/
logs/
proxy/
*.blend1
*.autosave
*.safetensors
*.gguf
*.ckpt
*.pt
*.pth
*.bin
```

LoRA 權重放在外接碟，sha256 記在 `lora/README.md`，不進 git。

### commit 規則

1. 一批一個 commit，訊息寫段落、步驟與原因，例如 `chorus1 video: 選定 s021–s034；s027 M 換 seed`。
2. 每道審查打一個 tag，例如 `g2-chorus1-r1`。
3. 看鏡頭歷史：`git log -- shots/s014/`。拿回舊版：`git restore --source=<hash> shots/s014/video.mp4`，然後再 commit。

### 可追溯性

GPU 有非確定性，目標是「可追溯、近似重現」，不保證逐位元相同。每次生成記錄：

- 模型與 LoRA 的檔名、sha256、LoRA strength；
- workflow 檔、ComfyUI commit、custom node snapshot（ComfyUI-Manager）；
- seed、sampler、scheduler、steps、cfg、shift、解析度、幀數；
- 輸入檔（關鍵幀、pose、音檔）的 sha256；
- `uv.lock`、torch、CUDA、驅動版本。

### `shotlist.json` schema（精簡版）

```json
{
  "schema_version": 1,
  "project": {"edit_fps": 24, "total_frames": 3223, "width": 1920, "height": 1080,
              "audio": {"path": "audio/song.wav", "sha256": "…"},
              "comfyui_commit": "…", "nodes_snapshot": "workflows/nodes_snapshot.json"},
  "shots": [{
    "shot_id": "s014", "section": "chorus1", "tier": "normal", "lyric_lines": [12, 13],
    "start_frame": 1152, "end_frame": 1205, "bars": [22, 22],
    "shot_size": "MS", "camera": "slow push-in", "route": "A", "status": "g3_review",
    "keyframe": {"file": "shots/s014/keyframe.png", "prompt": "…", "negative": "…", "seed": 1234,
                 "model": {"file": "Illustrious-XL-v1.0.safetensors", "sha256": "…"},
                 "loras": [{"file": "luna_illustrious.safetensors", "sha256": "…", "strength": 0.8}],
                 "workflow": "workflows/keyframe.json", "sampler": "euler_a", "steps": 30, "cfg": 6.0},
    "video": {"file": "shots/s014/video.mp4", "model": {"file": "…", "sha256": "…"},
              "workflow": "workflows/route_a.json", "seed": 482193, "steps": 20, "cfg": 5.0, "shift": 8.0,
              "gen_fps": 16, "gen_frames": 49, "trim_in": 4, "trim_out": 40,
              "candidates": [482193, 991, 5521], "selected": 482193},
    "review": [{"round": "r1", "gate": "G3", "code": "M", "row": 37}]
  }]
}
```

## Arch Linux 環境建置

沿用現有 ComfyUI（`~/creative/ComfyUI`，已是 Python 3.12、torch 2.11.0+cu130），只更新與補裝節點；allin1 另開 CPU 舊版環境。

### 系統套件

```bash
sudo pacman -S --needed git git-lfs ffmpeg uv nvtop kdenlive blender zram-generator earlyoom pacman-contrib
sudo systemctl enable --now earlyoom
# DaVinci Resolve 從 AUR 安裝
```

uv 在 extra 庫；`~/.local/bin/uv` 也有一份，注意 PATH 優先順序（本機）。

### ComfyUI

```bash
cd ~/creative/ComfyUI && git pull
source comfyui/bin/activate
uv pip install -r requirements.txt
python main.py --fast-disk --disable-pinned-memory --reserve-vram 1.5
# 真的 OOM 才加 --cache-none
```

- 更新前先用 ComfyUI-Manager 存一份 snapshot，更新後的 commit 記進 `shotlist.json`。
- 現有節點：comfyui\_controlnet\_aux（DWPose）、ComfyUI-Depthflow-Nodes、ComfyUI-Frame-Interpolation（RIFE）、ComfyUI-GGUF、ComfyUI-KJNodes、ComfyUI-VideoHelperSuite、rgthree-comfy（本機）。
- 要補裝：ComfyUI-Manager、ComfyUI-WanVideoWrapper（AniSora 用）、OpenPose 編輯器與姿勢插值節點（節點選擇需實測）。
- **不要裝 `ComfyUI-MiniMaxH3-Cache`**：它會 monkey-patch 模型程式碼，弄壞生成（查證）。
- 不要用 `--disable-smart-memory`：它會強制把模型積極 offload 到 RAM，反而增加 RAM 壓力（本機原始碼）。
- 不要用 `--lowvram`：在 dynamic VRAM 模式下沒有作用（本機原始碼）。
- SageAttention 目前沒裝；要加 `--use-sage-attention` 得先裝支援 sm\_120 的版本（需實測）。VAE 一律用 Tiled Decode。

### 音樂分析（獨立的 CPU 環境）

```bash
mkdir -p ~/creative/env-allin1 && cd ~/creative/env-allin1
uv venv -p 3.11 && source .venv/bin/activate
uv pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
# natten 0.17.5 的 CPU wheel，從 natten.org 舊版 wheel 索引安裝（不要裝 0.20 以後的版本）
uv pip install git+https://github.com/CPJKU/madmom
uv pip install allin1 librosa
allin1 ~/creative/Video_Generation/audio/song.wav -o struct -d cpu
```

allin1 用舊版 NATTEN API，0.20 以後被移除；0.17.5 沒有 cu128 版本，所以只能走 CPU（查證）。裝不起來就套用 allin1 尚未合併的 PR #39（純 PyTorch 實作）。

### LoRA 訓練

- Illustrious：kohya sd-scripts，獨立 uv 環境。
- Wan LoRA（試作判定需要時才做）：musubi-tuner，從 bf16 底模起訓（約 57GB 磁碟），只訓 low-noise expert、開 block swap。32GB RAM 很吃緊，列為最後手段。

## 資源與工時估算

用加速 LoRA 全片約 40–60 GPU 小時，完整步數約 100–150 小時；以下是試作前的估計（建議），P4 後用實測數字更新。

| 項目 | 估計 |
| --- | --- |
| 鏡頭數 | 45–55 鏡，其中需要影片模型的約 35–45 鏡 |
| 影片生成次數 | 每鏡平均 3–4 次，hero 鏡頭更多，共約 150–200 次 |
| 每次生成（Wan 14B、720p、81 幀、5060 Ti） | 完整 20 步約 45–90 分鐘；加速 LoRA 4 步約 8–15 分鐘（Q8 加 offload 的實際時間需實測） |
| 全片 GPU 時間 | 加速 LoRA 約 40–60 小時；完整步數約 100–150 小時 |
| 磁碟 | 新增權重（Q8 為主）與工作檔、候選，總量需實測 |

換了步數或解析度，同一個 seed 的結果就不同，所以「先用草稿挑 seed、再用正式設定重出」行不通；草稿只能確認構圖與動作方向。

## 風險與對策

最大的風險是主模型的畫質不符合你的要求；8 秒試作就是為了提早發現。

| 風險 | 內容 | 對策 |
| --- | --- | --- |
| 試作不過 | 主模型畫質不符合你的要求 | 換候選模型；或調整概念，增加視差、特效、剪輯的比重 |
| RAM 不足 | Q8：Wan 14B 雙 expert 各 15.4GB、Qwen-Edit Q8 21.8GB；H3 19.5GB＋14.6GB | 依序載入；`--fast-disk`；zram；仍不行就該步降到 Q6 |
| 磁碟不足 | 現在只剩 75G，Q8 權重較大 | 清快取、搬冷檔到外接碟；量產前從 Windows 切空間 |
| 參考圖生成失敗 | 整張角色表輸入時細節流失或角色走樣 | 加入裁切單格當第二張參考；仍不行退回裁切加去字 |
| 角色一致性 | 影片後半段漂移；模型間風格跳動 | 鏡頭 ≤ 2 小節；給尾幀；單一主模型；統一調色；必要時才訓練 Wan LoRA |
| 舞蹈與姿勢變形 | 動漫比例和真人動作資料不合 | Wan-Dancer 或 Animate 由模型內部處理體型；手動骨架照 LUNA 比例擺；舞蹈只做 3–6 個 hero 鏡頭 |
| 細節崩壞 | 手指、銀鏈、腿環容易糊或變形 | 構圖時避開；VACE 修補 |
| 補幀與放大 | 動漫線稿補幀有鬼影；逐幀放大會閃爍 | 試作時就測；必要時改用簡單放大加銳化 |
| allin1 安裝 | NATTEN 版本衝突 | CPU 舊版環境；PR #39 |
| LFS 額度 | 改寫歷史不會釋放空間 | 只 commit 選定版；候選不進 git |
| 品牌字標 | 模型寫不出正確的「CSFCCA」「LIVE AI」 | 後製疊加官方素材 |

## 授權

非商業、只給自己看的前提下，已查到的授權都允許使用；硬限制是 H3 的輸出不能拿來訓練其他模型，Illustrious 的條款全文尚未讀。

| 模型 | 授權重點 |
| --- | --- |
| MiniMax H3 | 排除美國、歐盟、英國、韓國，台灣個人使用不受影響；禁止用輸出訓練或改進其他模型（查證） |
| Wan 2.2 家族（I2V、Animate、Dancer、Fun Control、VACE-Fun）、Index-AniSora | Apache 2.0（查證） |
| Qwen-Image-Edit-2511 | Apache 2.0（查證） |
| Illustrious XL v1.0 | 模型頁標示「sdxl-license」，條款全文未讀（查證） |
| FLUX.2 Klein 9B（備案） | FLUX Non-Commercial License（查證） |
| HY-Motion 1.0（備案） | 排除歐盟、英國、韓國（查證） |
| Bailando（備案） | S-Lab 非商用（查證） |

日後若要公開發佈，需要重新檢查各模型條款，尤其是 H3。以上不是法律意見。

## 參考資料

- [agency-agents（msitarzewski）— GitHub](https://github.com/msitarzewski/agency-agents)
- [Qwen-Image-Edit-2511 — Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)；[Qwen Image Edit 2511 & Qwen Image Layered — Comfy 部落格](https://blog.comfy.org/p/qwen-image-edit-2511-and-qwen-image)；[Qwen Image Edit 2511 多參考圖 — RunComfy](https://www.runcomfy.com/comfyui-workflows/qwen-image-edit-2511-in-comfyui-precision-instruction-editing)；[Qwen-Image — GitHub](https://github.com/QwenLM/Qwen-Image)
- [CharacterSheet LoRA — Hugging Face](https://huggingface.co/Alissonerdx/CharacterSheet)（備案）；[FLUX.2 Klein — Black Forest Labs](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)（備案）
- [ComfyUI-SAM3 — RunComfy](https://www.runcomfy.com/comfyui-nodes/ComfyUI-SAM3)（備案）
- [MiniMax H3 in ComfyUI — ComfyUI Wiki](https://comfyui-wiki.com/tutorial/advanced/video/minimax/minimax-h3)；[MiniMax H3 License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Wan-AI — Hugging Face](https://huggingface.co/Wan-AI)；[Wan-Dancer-14B](https://huggingface.co/Wan-AI/Wan-Dancer-14B)；[Wan2.2 Animate ComfyUI 教學](https://docs.comfy.org/tutorials/video/wan/wan2-2-animate)
- [Wan2.2 Fun Control ComfyUI 教學](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control)；[Wan 2.2 Fun Control — sbcode](https://sbcode.net/genai/wan2.2-fun-control/)；[Wan2.2-Fun-A14B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control)
- [Wan2.2-VACE-Fun-A14B](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B)
- [Index-AniSora — bilibili](https://github.com/bilibili/Index-anisora)
- [Illustrious-XL-v1.0 — Hugging Face](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v1.0)；[Best local anime image model — LocalAIMaster](https://localaimaster.com/blog/best-local-anime-image-model)
- [kohya-ss sd-scripts](https://github.com/kohya-ss/sd-scripts)；[musubi-tuner](https://github.com/kohya-ss/musubi-tuner)
- [allin1](https://github.com/mir-aidj/all-in-one)；[issue #30](https://github.com/mir-aidj/all-in-one/issues/30)、[issue #35](https://github.com/mir-aidj/all-in-one/issues/35)、[PR #39](https://github.com/mir-aidj/all-in-one/pull/39)；[NATTEN 安裝](https://natten.org/install/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Depth Anything 3 — ComfyUI 教學](https://docs.comfy.org/tutorials/utility/depth-anything-3)
- [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0)；[ComfyUI-HyMotion](https://github.com/Aero-Ex/ComfyUI-HyMotion)（備案）
- [Motius-Bailando-AISTPP](https://huggingface.co/ZeyuLing/Motius-Bailando-AISTPP)（備案）
- [Git LFS billing — GitHub Docs](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)；[Removing files from Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage)
- 本機檢查：`df -h`、`du`、`free -g`、`swapon --show`、`nvidia-smi`、ComfyUI `comfy/cli_args.py`、`models/`、venv 套件版本、`動畫_女生_LUNA.jpg`
- 歌曲 BPM、能量與段落：librosa 對 `歌曲.mp3` 的初步分析
