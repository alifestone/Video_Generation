10_kneel_land_s0.png 的人物動作我希望可以類似 kneel_land_example.jpg 的動作。其他圖 ok

## Claude 回覆（2026-10-24）
- 10_kneel_land：照 kneel_land_example.jpg 重出。只寫 prompt（superhero landing）做不出來，改成 OpenPose 骨架（refs/poses/superhero_landing_portrait.png）＋參考圖輪廓線（superhero_landing_lineart.png）兩層 ControlNet；6 個 seed 挑腿環在跪地那條腿（她的右腿）的一張。舊圖在 v2/eval_r1/。
