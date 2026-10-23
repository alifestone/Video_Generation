# plan/：結構化劇本（P2 正本）

`plan/` 的 YAML 是劇本正本；`docs/` 與線上 doc 都由 `tools/render_plan.py` 產生，不手改。
格式與歌曲無關，下一支影片沿用同一套 schema，只換內容。

| 檔案 | 層級 | 內容 |
| --- | --- | --- |
| `project.yaml` | 參數 | 敘事類型、歌詞對應方式、角色與道具限制等，每支影片開頭由使用者選 |
| `glossary.yaml` | 名詞 | 詞彙（景別、轉場、後製效果、方向）＋實體（角色、場景、物件、特效、狀態） |
| `story.yaml` | 敘事 | 一句話概念、每段落發生什麼；只寫劇情，不寫畫面細節 |
| `shots.yaml` | 鏡頭 | 每一鏡的時間、主體位置、首幀、運動、末幀、轉場、拍點事件、後製 |

## 三條硬規則

1. **名詞先定義再引用**：story 與 shots 裡的角色、場景、物件、特效一律寫 `@id`，不得用別名；
   每個實體在 glossary 寫清楚外觀（形狀、大小、顏色、材質、是否發光）與「不是什麼」（`not_this`）。
2. **首幀與運動分開**：`first_frame` 是一張靜止畫面，不寫正在進行的動作；
   動作只寫在 `motion`，並分成 `camera`（鏡頭怎麼動）與 `subject`（畫面裡的東西怎麼動）。
3. **每一鏡都要有 `transition_in`**：類型只能用 glossary `vocab.transition` 的值；
   需要逐拍發生的事（倒數、閃動）寫成 `beat_events` 清單，一拍一筆，不寫「閃三次」。

## 時間位置

位置寫成 `[小節, 拍]`，拍從 1 起算，可寫小數（`2.5` = 第 2 拍後半）；`[0, 0]` 是歌曲開頭。
`span.to` 不含在本鏡內，等於下一鏡的 `span.from`。不規則小節（例如 3 拍、5 拍）照 `analysis/timeline.json`。

## 首幀、末幀

`first_frame` 是本鏡第一個完整畫面，不含轉場效果（淡入、閃白都疊在它上面）；`last_frame` 是本鏡最後一個完整畫面。
兩者都是靜止畫面，只寫「此刻看起來怎樣」，不寫正在發生的動作。`variation`（small／medium／large）決定要不要寫末幀，定義見 glossary。

## 方向

畫面上的左右一律是**觀眾看到的左右**（`screen_left`／`screen_right`）。
要指角色自己的左右時，寫 `char_left`／`char_right`，例如腿環在 `char_right` 大腿。

## 狀態

每個實體與鏡頭都有 `status`：`proposed`（Claude 提案，待確認）或 `confirmed`（使用者確認）。
`questions` 欄記錄還沒回答的問題；`tools/lint_plan.py` 會把 `proposed` 與 `questions` 全部列出。

## 流程

```text
project.yaml → glossary.yaml → story.yaml → shots.yaml
  → python3 tools/lint_plan.py          # 程式檢查（引用、別名、轉場、時間覆蓋、必填欄位、模糊詞）
  → 歧義審稿（tools/prompts/plan_critic.md，交給另一個 Claude 跑）→ 問題清單給使用者
  → python3 tools/render_plan.py        # 產生 docs/mv_plan.md 與線上 doc 的內容
```
