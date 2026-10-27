# P4 影片生成試作筆記（pilot/notes.md）

工具：`tools/p4_video.py --combo <組合> --shot <sNNN>`；數據見 `pilot/metrics.json`，log 在 `pilot/logs/`。
共同設定：prompt／negative／seed 20261027 取自 `pilot/prompts.yaml`；關鍵幀縮放涵蓋後中心裁切。Wan／AniSora 1280×720、81 幀、16fps、Q8 GGUF（未降解析度）；H3 因需 32 倍數以 1280×736 生成、124 幀 24fps，輸出時中心裁成 1280×720（音訊已去除）。
Wan 完整步數 shift 用 8（依任務指示；ComfyUI 範本實際預設為 5）。H3 範本沒有 CFG／negative（BasicGuider、res_multistep/simple）。

## 量測

| 組合 | 步數／cfg／shift | 單支時間 | ComfyUI RSS 峰值 | 系統 used／swap 峰值 | VRAM 峰值 |
|---|---|---|---|---|---|
| wan22_lightx2v | 4（2+2）／1／5 | 約 11 分（高噪 307s＋低噪 310s＋VAE 29s＋編碼 17s） | 25 GB | 22.5／5.9 GB | 10.2 GB |
| wan22_lightx2v_flf | 同上 | 11 分 | 24 GB | 20／6 GB | 10.2 GB |
| wan22_full | 20（10+10）／3.5／8 | 約 99 分（每步約 295s） | 25 GB | 20／8.5 GB | 10.3 GB |
| anisora（完整步數） | 同 wan22_full | 約 99 分 | 26.7 GB | 17.6／5.3 GB | 10.3 GB |
| anisora_lightx2v | 4／1／5 | 11.4 分 | 25.7 GB | 17.4／7.3 GB | 10.3 GB |
| h3_turbo | 8／無 CFG | 約 9.4 分（取樣 487s＋VAE 50s＋條件編碼 18s） | 26 GB | 29／10.5 GB | 14.7 GB |
| h3_full | 20／無 CFG | 約 21 分（取樣 1185s） | 27 GB | 28.5／5 GB | 14.9 GB |

- Q8 在 16GB VRAM＋32GB RAM 可跑，但 Wan 全程是 ComfyUI 自動 lowvram 部分載入（可用 VRAM 約 1.2GB，其餘權重在 RAM），所以完整步數特別慢；RSS 約 25GB、swap 用到 5–8 GB，沒有 OOM。H3 系統 used 峰值 29GB／swap 10GB，是最吃記憶體的，接近極限。
- 載入模型時間：ComfyUI 的模型實際載入是在取樣節點內惰性發生，無法單獨拆出；per_node_s 中「取樣」已含載入，metrics.json 有各節點耗時。
- 沒有發生錯誤，沒有降到 960×544。

## 評估（逐格圖在各資料夾 sNNN_sheet.png）

### wan22_lightx2v
- s003：角色與畫風穩定，LUNA 靜止。但 ECHO 滑入太早，之後橫越到 LUNA 左側再飄回右側，全程側面（面罩朝左的側視），沒有轉正；尾端停在 LUNA 右肩前方，不在右上，有拖影。
- s004：約 0.7–1.4 秒間睜眼（符合約 1 秒），睜眼自然，灰紫眼色與線條色塊乾淨，無崩壞。四組中 s004 最符合 prompt。

### wan22_lightx2v_flf（首尾幀）
- s003：ECHO 從右緣滑入、一開始就是正面朝鏡頭（沒有先朝左），中段有一格輕微變形（雙球輪廓），之後平順停在右上，最後一格接上 last.png。尾幀約束讓它轉正，但失去「先朝左行進再轉正」的設計。LUNA 穩定。

### wan22_full
- s003：ECHO 形狀崩壞，變成帶環的飛碟狀（球體加青色環），沒有轉正，也沒有面罩；LUNA 穩定。比 lightx2v 差。
- s004：約 2.1–2.9 秒才睜眼，比 prompt 慢很多；眼色偏灰（少了紫），線條乾淨。完整步數沒有比 lightx2v 好。

### anisora（完整步數，同 wan22_full 參數）
- s003：ECHO 設計跑掉（黑色圓盤面罩、底部青色環），而且越飄越大、向左飄到 LUNA 身上遮住她的腰，沒有轉正。LUNA 本體穩定，線條乾淨。
- s004：約 2.1–2.9 秒才睜眼；眼睛被重畫成更「標準動漫」的大高光眼，與關鍵幀的眼型不同（角色漂移），但線條與色塊很乾淨。
- 動漫線條／色塊：AniSora 的線條確實乾淨銳利、沒有糊，但在 ECHO 物件一致性與關鍵幀忠實度上不比 Wan 好，且有重畫臉部傾向。

### anisora_lightx2v（只跑 s004）
- 掛 lightx2v（為 Wan 蒸餾）可正常載入並生成，11.4 分，睜眼約 1.4–2.1 秒，眼型同樣被重畫成大高光眼，眼色偏藍紫，沒有崩壞。相容性可行但漂移與完整步數的 AniSora 類似。

### h3_turbo
- s003：目前最好。ECHO 在右緣外滑入，滑入時面罩與青色光帶朝左，之後減速轉正面朝鏡頭並停在右上微微浮動，只有一顆；LUNA 靜止，畫風不糊。最後一格接上 last.png。
- s004：約 1.4–3.5 秒緩慢睜眼，過程中睫毛與眼瞼有半透明疊影（像溶接而不是眼皮抬起），睜眼後穩定，眼色灰紫，最後一格接上 last.png。睜眼速度比 prompt 慢，疊影是缺點。

### h3_full
- s003：與 turbo 幾乎相同的好結果，動作略更平順；ECHO 朝左滑入並轉正停在右上，接上 last.png。
- s004：前 2.2 秒完全閉眼，約 2.5–3 秒突然睜開，睜眼瞬間有一兩格模糊疊影，之後穩定。時機比 prompt 晚；不比 turbo 好。

## 小結
- ECHO 轉正這種有時序的動作只有 H3（首尾幀）做得出來，turbo 與 full 品質接近，turbo 只要 9.4 分（full 21 分）。Wan 首幀版 ECHO 不轉正；Wan FLF 能轉正但沒有先朝左。
- s004 睜眼時機：Wan lightx2v 最接近約 1 秒；H3 與完整步數 Wan 偏慢（2–3 秒），可用 prompt 改寫或後製裁掉前段處理。
- 加速 LoRA：Wan lightx2v 在這兩鏡不比完整步數差（甚至更好）且快 9 倍；H3 turbo 與 full 相當。
- 記憶體：H3 最緊（系統 used 約 29GB、swap 10GB、VRAM 14.9GB），建議 H3 生成時不要同時跑其他東西。

---
# 第二輪（主管審查後）

## 第一輪審查補記
- wan22_lightx2v s004：睜眼乾淨、時機約 1 秒，但虹膜偏亮紫羅蘭，與 last.png 的灰紫不符（角色漂移）。
- h3_turbo s004：睜眼為半透明交叉溶接（閉眼／睜眼兩層疊），不是眼瞼動作，不可用（用密集 sheet 確認，見 h3_turbo/s004_open_sheet.png：前段睫毛淡出、灰色半透明眼球浮現）。
- h3_turbo s003：最佳，照 prompt 且接上 last.png。

## s004 睜眼生成（sNNN_open_sheet.png＝睜眼前後約 1 秒每 2 幀；另有精細逐幀 sheet）
| 組合 | 耗時 | VRAM／RAM 峰值 | 睜眼 | 溶接 | 眼色 |
|---|---|---|---|---|---|
| wan22_lightx2v_flf（首＋尾幀） | 11.2 分 | 10.3GB／RSS 26.8GB | 約 0.7–0.9 s（第 11→12 幀眼瞼一格內跳到半開，12–15 幀開完） | 無，逐幀確認無疊影 | 灰紫，與 last.png 相符 |
| h3_turbo_i2v（只首幀） | 8.5 分 | 14.7GB／used 28.5GB | 約 2.6–3.7 s，太晚 | 無溶接，但 61→62 幀硬切成半開，眼球內部亂碼狀、下眼瞼有雜點，之後 3 幀一階的階梯式動作，眼睛變小、灰無高光 | 灰，眼型漂移 |
| h3_turbo + nofade prompt（首尾幀，補測） | 9.3 分 | 14.7GB | 約 1.9–3 s（比原本早） | 仍有溶接（睫毛淡出、灰色半透明眼球），prompt 沒解掉 | 灰紫 |

- 1 號（Wan FLF）解掉了 wan22_lightx2v 的虹膜漂移：尾幀把眼色拉回灰紫，動作仍是真正的眼瞼抬起。缺點只有：睜眼是一格內跳開（16fps 下約 62ms，偏快，可用 RIFE 補）。目前 s004 最佳選擇。
- 2 號（H3 只首幀）不是溶接，但品質更差，不採用。結論：H3 的 s004 問題不在尾幀 guide，而是 H3 對眼瞼細節本身弱。
- 3 號：任務條件是「1 或 2 有溶接」；這兩支實際都沒有，所以依原意改在真正有溶接的 h3_turbo（首尾幀）上補測 prompt 強調 "eyelids physically lift open, no crossfade"（prompt 全文在 metrics.json prompt_used）。結果仍溶接，prompt 無效。
- 觀察：H3 在 s004 快速動作段呈現 3 幀一階的階梯（相鄰幀差尖峰每 3 幀一次，等效約 8fps）；s003 滑動段則連續。Wan 的 s004 睜眼也有每 2 幀一階（等效 8fps）。
- 建議：s004 用 wan22_lightx2v_flf；s003 用 h3_turbo。

## RIFE（16→48fps 再隔幀抽成 24fps）
- rife49，multiplier 3；s003 34 秒（含載入）、s004 8 秒；VRAM 輪詢只量到約 1GB（極輕）。輸出 post_tests/rife_s00{3,4}_24.mp4（另有 _48）。
- 逐幀看 ECHO 滑入段（rife_s003_echo_sheet.png，幀 10–21）與睜眼段（rife_s004_open_sheet.png，幀 20–37）：睜眼過程平順、無疊影；ECHO 沒有雙影，僅在 ECHO 與艙玻璃交疊的第 18 幀左側有輕微軟邊光暈。ECHO 的橫向拖影線是來源就有（Wan 原片）。
- 相鄰幀差：s003 最大 4.57→3.45、無尖峰；s004 最大 15.75→12.43，尖峰仍在睜眼區（24–36 幀）但沒有新增的異常尖峰，只是把 2 幀階梯拆成較細的階。
- 判斷：RIFE 可用，睜眼與滑入都無明顯鬼影；但它只讓運動更順，不會修掉原生動作的階梯感。

## Real-ESRGAN anime 6B（h3_turbo s003，124 幀）
- 整幀 fp16 不分塊可跑：每幀 0.795 s（第一幀 0.93 s），124 幀共 99 秒；torch 峰值 allocated 7.04GB、reserved 10.05GB。輸出 esrgan_s003_1080.mp4；對照 lanczos_s003_1080.mp4。
- 100% 裁切（crop100_lanczos_left_esrgan_right.png）：ESRGAN 線條明顯更銳利（臉、髮絲、鍊飾、ECHO 輪廓），lanczos 偏軟；ESRGAN 稍微過銳，ECHO 白色高光略被推到爆白。
- 閃爍（1080p 靜止區域，相鄰幀平均差）：LUNA 身體 ESRGAN 0.133／lanczos 0.106（+26%）；右背景 0.030／0.029；左背景 0.030／0.022（+36%）；亮度逐幀 std 都在 0.03–0.06（兩者相當，無整體亮度閃爍）。ESRGAN 略多一些細節抖動，絕對值很小，肉眼沒有明顯閃爍。
- 判斷：值得用，成本約每 5 秒鏡頭 1.7 分鐘，記憶體 10GB 以內；用前須確認 ComfyUI 空閒。

## s002 路線 C 視差
- 深度：DepthAnythingV2 vitl（controlnet_aux 下載 1.3GB，磁碟 165GB 餘，OK），深度圖 post_tests/s002_depth.png（牆遠、地板近、艙獨立一層，平滑合理）。未用 Depthflow（其預設動作為循環型，難精準控制到 5%）；改自寫 torch 反向取樣視差 tools/p4_parallax.py。
- 輸出 pilot/s002_parallax.mp4：1920×1080、54 幀＠24fps＝2.25 s，緩入緩出推近平均 5%（遠 4.3%、近 7.3%）。
- 檢查：螢幕格線、艙邊緣、地板格線無撕裂、無摩爾閃爍（相鄰幀差平滑，最大 1.37 vs 中位 1.02，無尖峰）；對照圖 s002_parallax_crops_first_left_last_right.png。相對均勻 5% 放大，牆左右約 2.3 px 向外、地板約 2.8 px 垂直位移，是預期的視差，沒有扭曲；艙底與地板格線在艙邊有輕微拉伸，不明顯。
- 判斷：路線 C 可行。keyframe 原生 1344×768，輸出 1920×1080 為放大（可再配 ESRGAN）；若要更明顯的視差可提高 --spread。

## 新增／修改檔案
tools/p4_video.py（新增 h3_turbo_i2v 組合與 --variant nofade）、tools/p4_analyze.py（幀差序列、尖峰、sheet）、tools/p4_post.py（rife／esrgan）、tools/p4_parallax.py。
