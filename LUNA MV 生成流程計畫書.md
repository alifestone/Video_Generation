# LUNA MV 影片生成流程計畫書

Oct 14, 2026 · @周聖詠

## 前言與原則

本計畫規劃「影片怎麼生成」的流程，並且本計畫**不沿用任何先前的記憶、先前的生成結果或先前的工作流程設定**，因為先前生成的影片都不如預期。

前提條件：

- **不需要對嘴**：LUNA 不唱歌，所有 lip-sync 工具都排除在外。
- **非商業用途，影片只給自己看**：可以使用非商用授權的模型。
- **不以時間為限制**：流程以品質優先，可以選慢但好的設定（例如 Q8 GGUF）。
- **版本追蹤用 git**：遠端倉庫 `git@github.com:alifestone/Video_Generation.git`，不用檔名記版本。
- **硬體與作業系統**：

| 項目 | 規格 | 對流程的影響 |
| --- | --- | --- |
| CPU | Ryzen 7 7700（8C/16T） | 音樂分析、ffmpeg、補幀的 CPU 部分夠用 |
| GPU | RTX 5060 Ti 16GB（Blackwell，sm\_120） | 一次只能常駐一個大模型；需 CUDA 12.8 以上的 PyTorch |
| RAM | 32GB DDR5 | 模型 offload 的上限；Q8 與 H3 都會用到接近上限 |
| 儲存 | 1TB NVMe Gen4 | 模型權重約 150–250GB |
| OS | Arch Linux（Hyprland／Wayland） | 每個工具用 uv 建獨立 Python 環境 |

本文件標註「查證」的內容來自 2026 年 10 月的網路資料，標註「建議」的是規劃判斷。

## 素材盤點

歌曲長 2 分 14.3 秒，約 108 BPM（每拍 0.557 秒、每小節約 2.23 秒），結構是「低能量開場 → 副歌式 hook → 主歌 → 副歌 → 英文橋段 → 最強的終場副歌」。這些數字是我用 librosa 對你上傳的音檔初步分析的結果，正式流程由 allin1 加 librosa 產出。

| 素材 | 狀態 | 需要的處理 |
| --- | --- | --- |
| 音檔（上傳的是 mp3，不是 mp4） | 134.3 秒、44.1kHz 立體聲、128kbps；檔案內嵌「AI 生成內容」標記 | 先轉成 WAV 再分析（allin1 官方說不同 MP3 解碼器會偏移 20–40 ms） |
| 歌詞 SRT | 32 行，0:00.68–2:12.22；斷句已對齊人聲 | 與拍點、段落合併成主時間軸 |
| 歌詞純文字 | 與 SRT 內容大致相同，部分字詞不一（如「訊息／訊號」「指尖／之間」） | **以 SRT 為準** |
| LUNA 角色表 | 單張 1920×1080，含全身站姿、4 張動作角度、9 種表情、個性說明 | 由 Claude Sonnet 5.5 透過 GIMP MCP 裁成獨立參考圖；去掉上方網站導覽列與左側「LIVE AI LUNA」直寫字 |

初步偵測到的段落邊界與能量（以全曲最大音量為 10）：

| 時間 | 能量 | 對應歌詞（SRT 行號） | 推測段落 |
| --- | --- | --- | --- |
| 0:00–0:12 | 4 | 1–2「訊號亮起…世界重啟」 | Intro |
| 0:12–0:20 | 8 | 3「321 LIVE AI」 | Hook |
| 0:20–0:37 | 6 | 4–6 | Verse |
| 0:37–0:48 | 6–7 | 7–11 | Pre-chorus |
| 0:48–1:17 | 7–8 | 12–18 | Chorus 1 |
| 1:17–1:25 | 7–8 | 19–20 英文段 | Bridge |
| 1:25–1:33 | 6–7 | 21 | Build（能量短暫下降） |
| 1:33–2:10 | 9–10 | 22–31 | Final chorus（全曲最高點） |
| 2:10–2:14 | 5→0 | 32「CSFCCA LIVE AI」 | Outro |

兩點注意：電子曲的節拍偵測常有倍速誤差，實際可能是 108 或 216 BPM，由 allin1 的首拍（downbeat）輸出確認。音檔本身沒有影片畫面，整支 MV 要從零生成。

## 修正後的完整流程

修正後有兩道人工審查：第一道在關鍵幀，第二道在影片。先在關鍵幀階段把構圖、角色、色調定下來，影片階段只需處理動作和穩定度。

&#91;embedded content: 修正後流程 · 8 個步驟、2 道人工審查\]

鏡頭清單 `shotlist.json` 是全流程的主檔：每個鏡頭的時間碼、生成路線、動作來源、prompt、seed 都記在這裡；版本由 git 管理。

| # | 步驟 | 輸出 | 工具 | 驗收標準 |
| --- | --- | --- | --- | --- |
| 0 | 素材整理 | `refs/`：從角色表裁出的單一參考圖（全身、4 張動作、9 種表情） | Claude Sonnet 5.5 加 GIMP MCP（`crop_to_rect`、`export_image`） | 每張只有一個角色、沒有文字 |
| 1 | 音樂分析 | `timeline.json`：BPM、拍點、首拍、段落、能量曲線、歌詞行 | allin1 加 librosa | 段落邊界和聽感一致，誤差小於 1 拍 |
| 2 | 風格錨定 | `style_bible.md`（外觀清單、色票、光影、場景、禁止事項）、`mv_plan.md`（MV 段落規劃與視覺概念）、風格樣張 | Claude Opus 5.5 加 agency-agents Visual Storyteller、agency-agents Brand Guardian；Qwen-Image-Edit 出樣張 | 你確認樣張與段落規劃 |
| 3 | 角色 LoRA | `luna_illustrious.safetensors`（關鍵幀用）；影片走樣時加訓 Wan 2.2 LoRA | kohya\_ss sd-scripts；Wan LoRA 用 musubi-tuner | 10 張測試圖的臉、髮型、服裝細節一致 |
| 4 | 劇本 | `script.md`：依 `mv_plan.md` 寫每段的敘事與情緒 | Claude Opus 5.5 加 agency-agents Visual Storyteller | 每句歌詞都有對應畫面 |
| 5 | 鏡頭清單 | `shotlist.json`：shot\_id、起訖時間、景別、運鏡、route、motion\_source、prompt、negative、seed | Claude 加 agency-agents Image Prompt Engineer；PenShot 選用 | 鏡頭總長等於 134.3 秒，切點都落在拍點上 |
| 6 | 關鍵幀 | 每個鏡頭的首幀（需要時加尾幀） | Qwen-Image-Edit-2511 GGUF；Illustrious 加 LUNA LoRA | 審查一通過 |
| 7 | 動作指導 | 路線 B 鏡頭的 OpenPose 骨架影片 | 見下方「動作指導來源」 | 骨架長度等於鏡頭長度，重拍動作落在拍點 |
| 8 | 影片生成 | 每個鏡頭一個 mp4 | 依路線選模型（見下表） | 審查二通過 |
| 9 | 局部修補 | 修好的片段 | Wan VACE inpainting 加 SAM 遮罩 | 瑕疵消失、周圍不變 |
| 10 | 後製合成 | `final.mp4` | RIFE 補幀、Real-ESRGAN anime 放大、ffmpeg、Kdenlive 或 DaVinci Resolve | 全片無跳幀、切點對拍 |

影片生成分三條路線，寫在每個鏡頭的 `route` 欄位：

| 路線 | 適用鏡頭 | 工具 | 本機需求（查證） |
| --- | --- | --- | --- |
| A 運鏡與氛圍 | 輕微動作、推拉搖移、場景特效 | MiniMax H3 I2V（可加尾幀） | 剪枝 INT8 19.5GB、NVFP4 文字編碼器 14.6GB、影片 VAE 4.9GB，需 block swap；最長 15 秒、短邊 768px |
| B 指定動作 | 舞蹈、比 V、指向鏡頭、轉身等要精準控制的動作 | Wan 2.2 Fun Control **Q8 GGUF**（首幀加骨架影片） | high／low noise 兩個模型各 15.4GB、umt5 Q8 6GB；單個就接近 16GB VRAM，需部分 offload |
| C 靜態插畫 | 歌詞字卡、快切節奏段、角色特寫 | 關鍵幀加 2.5D 視差運鏡（Depth Anything V2 深度圖） | 幾乎不吃 VRAM |

動作指導來源（不用真人驅動，全部由 AI 生成）：

| 來源 | 適用動作 | 工具 | 轉成骨架影片的方式 |
| --- | --- | --- | --- |
| 音樂生成舞蹈 | 副歌等跟著節拍跳舞的鏡頭 | Bailando（Motius 版，AIST++ 訓練；S-Lab 非商用授權）：輸入音訊，輸出 SMPL-24 關節、每秒 60 幀 | 寫腳本把 3D 關節投影成 2D OpenPose 骨架，重取樣成 24fps（建議） |
| 文字生成動作 | 比 V、指向鏡頭、轉身等描述得出來的動作 | HY-Motion 1.0（騰訊，2025 年 12 月開源；ComfyUI-HyMotion 節點可輸出 FBX、重定向到 Mixamo 骨架） | FBX 匯入 Blender，對齊鏡頭角度後算繪灰模人偶，再用 DWPose 擷取（建議） |
| 影片模型生替身 | 前兩種都不像時 | Wan 2.2 T2V 生成一段簡單背景、動作清楚的替身舞者影片 | DWPose 擷取骨架（建議） |
| 手動關鍵姿勢 | 很短或很精確的姿勢 | ComfyUI OpenPose 編輯器擺關鍵姿勢，姿勢插值節點補中間幀 | 直接就是骨架（需實測） |

骨架要先調成 LUNA 的體型比例（動漫角色腿較長、頭較大），鏡頭角度也要和關鍵幀一致，否則 Fun Control 會把角色拉變形（建議）。H3 會同時生成音訊；ComfyUI 不接 `VAEDecodeAudio` 就是靜音輸出，後製一律用原曲。

## 分級 Human review 與 git 版本追蹤

審查不通過時，依問題類型只退回到「能修好它的最後一步」，而不是每次都回到劇本。你只需要在 `review.csv` 填一個問題代碼和一句說明，Claude 就能判斷要重跑哪些鏡頭、哪些步驟。

| 代碼 | 問題類型 | 例子 | 退回到 | 重跑範圍 |
| --- | --- | --- | --- | --- |
| S | 敘事或段落概念 | 「副歌太平淡」「這段和歌詞無關」 | 劇本 | 只重跑該段落的劇本、鏡頭、關鍵幀、影片 |
| T | 節奏或切點 | 「這個鏡頭太長」「沒有對到鼓點」 | 鏡頭清單 | 先試後製重剪；不夠再重生該鏡頭 |
| K | 構圖、場景、色調 | 「背景不對」「太暗」 | 關鍵幀 | 該鏡頭的關鍵幀與影片 |
| C | 角色走樣 | 「臉不像 LUNA」「髮色變了」 | 關鍵幀；多個鏡頭都走樣時回 LoRA | 該鏡頭；若重訓 LoRA，後續鏡頭全部重出關鍵幀 |
| M | 動作或穩定性 | 「動作歪掉」「畫面闃爍」 | 影片生成 | 同一關鍵幀換 seed 或換路線重生 |
| L | 局部瑕疵 | 「手指多一根」「腰帶鎖鏈糊掉」 | 局部修補 | 只用 VACE inpainting 修該區域 |
| P | 畫質或流暢度 | 「太糊」「卡頓」 | 後製 | 放大或補幀參數 |

版本全部交給 git，檔名固定不帶版本號。重新生成就是覆蓋同一個檔案再 commit，舊版留在 git 歷史裡。遠端倉庫：`git@github.com:alifestone/Video_Generation.git`。

```text
Video_Generation/
├─ .gitattributes        # 圖片、影片、音訊走 Git LFS
├─ .gitignore            # ComfyUI 原始輸出、模型權重
├─ refs/                 # 裁切後的參考圖
├─ audio/                # song.wav、lyrics.srt
├─ analysis/timeline.json
├─ docs/                 # style_bible.md、mv_plan.md、script.md
├─ shotlist.json
├─ review.csv
├─ workflows/            # ComfyUI workflow（API 格式 JSON）
├─ lora/                 # 訓練設定與資料集說明
├─ shots/s14/            # keyframe.png、pose.mp4、video.mp4（固定檔名）
└─ final/
```

使用規則（建議）：

1. **一次重生一個 commit**，訊息寫鏡頭、步驟與原因，例如 `s14 video: M 換 seed 482193`。
2. **每輪審查打一個 tag**（`review-r1`、`review-r2`），`review.csv` 記錄審查的 commit hash、結果、問題代碼、說明。
3. 看某個鏡頭的歷史用 `git log -- shots/s14/`；要拿回舊版用 `git restore --source=<hash> shots/s14/video.mp4`。
4. `shotlist.json` 記錄每個鏡頭的 seed、模型、workflow 檔，配合 git 歷史就能完全重現任一版。

Git LFS 的額度要留意（查證）：GitHub 個人帳號免費提供 10 GiB LFS 儲存與 10 GiB 流量，每個舊版本都會佔空間；超過且沒有付款方式時就不能再 push 新檔。對策：只 commit 你要審查的候選版本，ComfyUI 的原始批次輸出不進 git；模型權重永遠不進 git。

```bash
sudo pacman -S --needed git-lfs
git clone git@github.com:alifestone/Video_Generation.git ~/mv/Video_Generation
cd ~/mv/Video_Generation
git lfs install
git lfs track "*.png" "*.jpg" "*.mp4" "*.wav" "*.mp3"
git add .gitattributes && git commit -m "Track media with Git LFS"
```

審查時檢查這五項：臉部是否像 LUNA、紫色編髮與銀鏈、白色高領拉鍊上衣、左大腿黑色腿環、手部手指數。

## 資源排程與 Arch Linux 環境

硬體上的鐵則是**同一時間只跑一個大模型**，所以流程依步驟整批跑：先把所有鏡頭的關鍵幀出完，再切換到影片模型整批生成。Claude 跑在雲端不佔 VRAM；如果 PenShot 接本機 Ollama，就必須在跑 ComfyUI 之前關掉。

| 模型 | 檔案大小（查證） | 16GB VRAM 上的做法 | 32GB RAM 風險 |
| --- | --- | --- | --- |
| Illustrious XL（SDXL） | fp16 約 6.9GB | 直接跑 | 低 |
| Qwen-Image-Edit-2511 | 有 Q8／Q5／Q4 GGUF；文字編碼器 Qwen2.5-VL 7B fp8 | 品質優先可試 Q8，放不下再降 Q5 | 中 |
| MiniMax H3（I2V） | 剪枝 INT8 19.5GB；NVFP4 文字編碼器 14.6GB；影片 VAE 4.9GB | block swap；文字編碼完就卸載 | **高**：19.5GB 權重放在 RAM 後所剩無幾 |
| Wan 2.2 Fun Control Q8 | high／low noise 各 15.4GB；umt5 Q8 6GB | 兩個 expert 輪流載入，由 ComfyUI 部分 offload | **高**：兩個加起來 30.8GB，若都留在 RAM 會超過上限 |
| Wan VACE（修補） | 14B，可用 GGUF | Q5–Q8 依實測 | 中 |
| HY-Motion 1.0／Bailando | 動作模型遠小於影片模型（HY-Motion 約 10 億參數，另有 Lite 版） | 直接跑 | 低 |

幾個必做設定：

- **Blackwell 需要 CUDA 12.8 以上的 PyTorch**（`cu128` 或更新的 wheel），舊版 wheel 不支援 sm\_120（既有知識）。
- **開一個 32GB 的 swapfile**：Q8 與 H3 都會把 RAM 吃到接近 32GB，swap 可以避免被 OOM killer 殺掉（建議）。
- **跑 Q8 時讓 ComfyUI 不快取模型**：改用 `--cache-none` 或 `--disable-smart-memory` 啟動，換取較低的 RAM 占用（既有知識，需實測哪個有效）。
- ComfyUI 需 0.30.0 以上才有 H3 節點；官方指南說 0.30.x 若載入很慢，可加 `--disable-pinned-memory`；**不要裝 `ComfyUI-MiniMaxH3-Cache`**，它會弄壞生成。
- 模型權重全部約需 150–250GB。

Arch Linux 上的環境規劃（建議）：Arch 的系統 Python 版本很新，很多 ML 套件還沒跟上，所以每個工具都用 uv 建 Python 3.12 的獨立環境。GIMP MCP 需要 GIMP 3.2 以上（查證）。

```bash
# 系統套件
sudo pacman -S --needed git git-lfs ffmpeg uv nvtop gimp blender

# GIMP MCP（插件放進 GIMP 使用者 plug-ins 資料夾，在 GIMP 內由 Tools > MCP > Start MCP Server 啟動）
git clone https://github.com/maorcc/gimp-mcp ~/mv/gimp-mcp && cd ~/mv/gimp-mcp && uv sync
claude mcp add gimp-mcp -- uv run --directory ~/mv/gimp-mcp gimp_mcp_server.py

# ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI ~/mv/ComfyUI
cd ~/mv/ComfyUI && uv venv -p 3.12 && source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
# custom nodes：ComfyUI-Manager、ComfyUI-GGUF、comfyui_controlnet_aux（DWPose）、ComfyUI-VideoHelperSuite、ComfyUI-HyMotion

# 音樂分析（allin1 + librosa）
mkdir -p ~/mv/analysis && cd ~/mv/analysis && uv venv -p 3.12 && source .venv/bin/activate
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# NATTEN 依 natten 官網對應 torch 版本的指令安裝
uv pip install git+https://github.com/CPJKU/madmom
uv pip install allin1 librosa
ffmpeg -i song.mp3 -ar 44100 song.wav
allin1 song.wav -o struct        # GPU 失敗就加 -d cpu

# 音樂生成舞蹈（Bailando，獨立環境；安裝方式依 Motius 模型卡）
```

PenShot 文件說 Redis 是選用的，沒有會退回記憶體加檔案儲存，所以單機使用不必裝。

## 風險與限制

技術上最大的風險是 **32GB RAM 撐不住 Q8 與 H3 的 offload**；流程上最大的風險是 **AI 生成的骨架比例與 LUNA 不合**，導致路線 B 的角色變形。

| 風險 | 內容 | 對策 |
| --- | --- | --- |
| RAM 不足 | Q8 兩個 expert 共 30.8GB、H3 剪枝模型 19.5GB，都接近 32GB RAM 上限 | 32GB swapfile；ComfyUI 關閉模型快取；仍不行就該步降到 Q6 |
| 骨架比例不合 | Bailando、HY-Motion 的動作來自真人體型，和動漫比例不同；Bailando 的舞風侷限於 AIST++ 的舞種 | 骨架先縮放到 LUNA 比例；先用 1 個鏡頭驗證從動作到影片的整段流程 |
| 開源影片模型的品質上限 | 手部、銀鏈、腿環等細節容易糊或變形；快速大動作易崩 | 細節特寫走路線 C；局部用 VACE 修補 |
| 角色一致性 | 影片模型只吃首幀時，後半段會漂移 | 鏡頭控制在 5 秒內；給尾幀；走樣多時加訓 Wan LoRA |
| allin1 相容性 | NATTEN 版本與 RTX 50 系列有未解 issue（查證） | 改用 CPU 模式 |
| GIMP MCP | 需 GIMP 3.2 以上；README 列出「復原／回滚」還在未來功能清單（查證） | 裁切在副本上做，原圖先 commit 進 git |
| Git LFS 額度 | 免費 10 GiB 儲存，每個舊版都算（查證） | 只 commit 審查用候選版；用量接近上限時再決定加購 |
| 品牌字標 | 影像模型寫「CSFCCA」「LIVE AI」文字常常走樣 | 字標一律在後製疊加官方素材 |

授權方面，本計畫非商業、影片只給自己看：MiniMax H3 的授權排除美國、歐盟、英國、韓國，在台灣個人使用不受影響；Bailando 為 S-Lab 非商用授權，符合本計畫用途。日後若要公開發佈，需重新檢查 H3 授權與各模型條款。以上不是法律意見。

## Sources

- [gimp-mcp（maorcc）— GitHub](https://github.com/maorcc/gimp-mcp)
- [agency-agents（msitarzewski）— GitHub](https://github.com/msitarzewski/agency-agents)
- [All-In-One Music Structure Analyzer（allin1）— GitHub](https://github.com/mir-aidj/all-in-one)；[allin1 issues](https://github.com/mir-aidj/all-in-one/issues)
- [penshot — PyPI](https://pypi.org/project/penshot/)；[PenShot 官方文件](https://shot.helpenx.com/docs/)
- [MiniMax H3 in ComfyUI — ComfyUI Wiki](https://comfyui-wiki.com/tutorial/advanced/video/minimax/minimax-h3)
- [MiniMax H3 Community License — Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Wan 2.2 Fun Control — sbcode](https://sbcode.net/genai/wan2.2-fun-control/)
- [Wan2.1-VACE native support — Comfy blog](https://blog.comfy.org/p/wan21-vace-native-support-and-ace)
- [HY-Motion 1.0 開源報導 — NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/tencent-open-sources-hy-motion-1-0-a-billion-parameter-text-to-motion-ai-model/)；[ComfyUI-HyMotion — RunComfy](https://www.runcomfy.com/comfyui-nodes/ComfyUI-HyMotion)
- [Motius-Bailando-AISTPP — Hugging Face](https://huggingface.co/ZeyuLing/Motius-Bailando-AISTPP)
- [How to Use Qwen-Image-Edit-2511 GGUF in ComfyUI — Kombitz](https://www.kombitz.com/2025/12/23/how-to-use-qwen-image-edit-2511-gguf-in-comfyui/)
- [Best local anime image model — LocalAIMaster](https://localaimaster.com/blog/best-local-anime-image-model)
- [Git LFS billing — GitHub Docs](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- 音檔 BPM、能量與段落：以 librosa 對上傳音檔實測
