# refs/entities/：非角色實體的固定參考圖（P3 步驟 6）

由 `tools/p3_entity_refs.py` 產生（Qwen-Image-Edit-2511 Q8＋Lightning 4 步，每張約 20–30 秒）。prompt 依 `plan/glossary.yaml` 各實體的 look 改寫成英文；每張的完整 prompt、seed、來源圖記在 `results.json`。
glossary 各實體的 `refs` 指向這裡。被淘汰的版本放在 `../entities_r0/`（不進 git）。

| 實體 | 保留 | 備註 |
| --- | --- | --- |
| `drone`（ECHO） | front、back | 三個側角度（three_quarter、side、night）兩輪都淘汰：文生圖會在光帶上多畫鏡頭或眼睛、外殼上出現類似文字的記號；從 front 改圖則球體轉不過去、光帶變形，夜景版外殼變成半透明。front 的黑色面罩 G0 使用者選保留（只在正面約 1/3 圈，見 glossary drone.look.visor） |
| `pod`（啟動艙） | closed、open、platform | open 第一、二輪玻璃罩都沒有照 glossary 分成兩半，G0（使用者選）改成從 closed 改圖：兩半往兩側滑開；底座頂面在 open 裡沒有發光，關鍵幀要補。platform 側面多一個圓孔，正式關鍵幀要再修 |
| `hq_control` | wide | |
| `monitor_wall` | color、frozen、city_1–7 | city_1–7 是監視牆各螢幕要貼的彩色城市畫面；第一輪有 4 張畫出螢幕外框，改 prompt 重出 |
| `launch_tunnel` | shaft | 第一輪有數字字樣與棋盤格破圖；第二輪畫成塔的外觀；第三輪改寫成「從井底往上看」才對 |
| `launch_roof` | hatch | |
| `street` | restored、frozen | restored 第一輪懸浮車貼地，改 prompt 後浮在約二樓高 |
| `hovercar` | side、three_quarter | |
| `signal_tower` | full_night、deck、base、control_offline、control_online | control_offline 的光柱仍有微光（glossary：斷電時不亮），關鍵幀要調暗 |
| `balloon_child`、`balloon` | back | 同一張 |
