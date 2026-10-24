- 01_intro_pod_s0.png 是哪個畫面的圖？它的背景畫風和 luna 的風格不同
- 02_hook_rooftop_s1.png 為什麼會有兩個 luna
- 03_verse_street 的兩張圖中的水灘請畫多一點，因為下雨不可能只有一個水灘
- 04_prechorus_cars_s1.png 的車子畫歪了
- 05_chorus1_burst_s0.png 的 luna 為什麼飄在空中
- 06_chorus1_climb 的兩張圖想要的是爬上塔嗎？如果是的話，luna 應該要面向塔，並做出向上爬的動作，另外 s0 為什麼會有光從 luna 的位置噴射出來

## Claude 回覆（2026-10-24）
01–06 改成對應到具體鏡頭，檔名帶鏡號；舊圖在 refs/style_samples_r0/。
- 01（原 intro_pod）：本來是 s003（她在啟動艙裡閉眼）。背景畫風不同是因為「glass capsule」讓底模畫成 3D 感的室內；改寫成「glass tube, stasis pod」後艙與背景都是平塗動畫風。
- 02 兩個 LUNA：底模在寬畫面偶爾會畫兩個人（負面詞已有 multiple girls 仍會發生），是壞圖。改成 s008 落地，用和 kneel_land 同一套姿勢控制。
- 03 水灘：改成 s010，prompt 加了 many puddles, puddles everywhere。
- 04 車歪：改成 s017。底模只用文字畫不出懸浮車（會畫成有輪子的普通車或乾脆不畫），見 G0 下一題。
- 05 為什麼飄在空中：這是 s024（Chorus 1 的 drop）。劇本裡 s023 她從車頂往前撲出時被灰色凍在半空，s024 顏色從胸口炸開時她還在半空，所以飄著是對的；但原本的姿勢（張開雙臂）不對，劇本是身體前傾、雙手往前伸。重出後爆發效果仍弱，見 G0 下一題。
- 06 爬塔：是 s029（蹬肋跳上維修平台）。改成背對鏡頭、面向塔、往上伸手蹬牆；s0 從她身上噴出的光是 prompt 裡「cyan glowing lines」被畫成光束，已拿掉 glowing。
