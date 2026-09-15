# /kvc — KV 快取相容的上下文壓縮(繁體中文說明)

> 對應程式:`kvc.ts`(pi 全域擴展)。本檔為說明文件,不影響 pi 載入。

## 一、解決什麼問題

pi 內建的 `/compact` 會把上下文壓縮成一份摘要,但它發送摘要請求的方式是:

- 用**另一個 system prompt**(專用的「摘要助手」提示詞)
- 把整段對話**重新序列化**成文字,塞進一條 user message
- 甚至帶新的 session id、關閉快取保留

結果:token 序列從第 0 個位置就和之前任何一次請求不同。對雲端 API 沒關係,
但在 **llama.cpp / LM Studio** 這種本地推理後端上,KV 快取完全命中不了,
必須把**整個上下文重新 prefill**——实测 41k tokens 要約 2 分鐘,
上下文越大(100k、200k+)越痛苦。

## 二、核心原理:前綴擴展(prefix extension)

llama.cpp(含 LM Studio 後端)的 KV 快取規則:**跨請求重用「共同前綴」**。
只要兩次請求的前 N 個 token 完全相同,第二次就只需 prefill 多出來的部分,
而且伺服器會保留多個已快取的前綴(快取池),不會互相頂掉。

`/kvc` 就是利用這一點,把摘要請求構造成「**上一次的正常請求 + 一條追加訊息**」:

```
[ 捕獲到的 payload:system prompt + 完整對話 + tools,與上次實際發出的逐字元相同 ]
        +
[ 一條追加的 user message:摘要指令(約 1k tokens) ]
```

這樣伺服器只需 prefill 最後那約 1k tokens 的指令,其餘全部命中 KV 快取。
壓縮完成後,下一次正常請求也只需 prefill「小的摘要 + 保留的近期訊息」。

實作上的關鍵細節:

1. **直接複用上次實際發出的 payload**。擴展掛在 `before_provider_request` 事件上,
   捕獲最後一個 OpenAI chat-completions 請求(跳過內建摘要請求本身)。
   不自己重建——因為任何重建(哪怕換個字)都會讓前綴失效。
2. **tools 必須保留**。llama.cpp/LM Studio 的 chat template 會把 tools JSON
   渲染進 prompt,拿掉 tools 前綴就變了。複用原 payload 自然解決。
3. 只改兩個不影響 prompt 的欄位:`stream: false`、`max_tokens`(設為摘要預算),
   並移除 `stream_options`。

## 三、為什麼不會把 system prompt 放兩次

壓缩後的上下文 = `system prompt + 摘要`。如果摘要內容又把 system prompt
抄一遍,就會出現「同一份設定在一個 context 裡出現兩次」的問題。

兩層處理:

1. **指令層**:追加的摘要指令明確要求——「頂部的 system prompt、工具定義、
   守則是**常駐配置**,會自動繼續生效,**不要**引述、改寫或摘要它們,
   輸出只描述對話本身」。實測模型遵守良好。
2. **後處理保險**:若模型仍把 system prompt 逐字複製在摘要開頭,
   程式會偵測(摘要與 system prompt 共享 100+ 字元的開頭)並裁掉重複部分。
   合法的摘要以 `## Goal` 開頭,不可能誤傷。

## 四、使用方式

| 指令 | 說明 |
|---|---|
| `/kvc` | 壓縮(行為等同 `/compact`,但走 KV 快取) |
| `/kvc <重點提示>` | 帶重點,例如 `/kvc 保留最近測試失敗的訊息` |

前提:目前 session 至少要完成過**一個 agent turn**(需要已捕獲的請求作為字首)。
新 session 直接 `/kvc` 會提示改用 `/compact`。

### 自動壓縮(85% 觸發,預設開啟)

長 session 不必記得打 `/kvc`:當上下文用到**視窗的 85%** 時,
擴充套件自動 arm 並走同一條 KV-cache-compatible 壓縮路徑——不會全量重填。
實作完全對照 pi 內建 auto-compact 的設計:

- **檢查時機**:每次 agent run 完全 settled(重試、排隊訊息都處理完)之後,
  用 pi 自己的計數(`getContextUsage()`,與 footer 顯示的百分比相同)判斷。
- **也接管 pi 內建的 threshold/overflow 自動壓縮**:小視窗下
  (`contextWindow - reserveTokens` 比 85% 更早觸發),自動壓縮同樣走 KV-cache 路徑。
- **回退規則與 `/kvc` 完全相同**:無捕獲、模型變更、非 OpenAI 相容 provider →
  改跑內建壓縮;手動 `/compact` 永遠保持內建行為(要快就用 `/kvc`)。

開關設定在 `~/.pi/agent/settings.json`(全域)或 `<project>/.pi/settings.json`
(專案優先),**預設開啟**:

```json
{ "kvc": { "autoCompact": false } }
```

自動觸發的決定(`TRIGGER` / `SKIP` 及原因)會記錄在暫存目錄的 `kvc-debug.log`。

## 五、安全與回退設計

### 為什麼 armed 標誌只「2 分鐘有效」

`/kvc` 的執行流程是「**先設 flag,再呼叫 `ctx.compact()`**,等 pi 觸發
`session_before_compact` 事件時,handler 靠這個 flag 判斷該走哪條路徑」。
這個事件 handler 是全域共用的——內建 `/compact`、自動壓縮、`/kvc` 都會觸發它,
唯一的區分方式就是這個 flag。

邊界情況:`ctx.compact()` 之後,pi 會先執行 `prepareCompaction()`;
如果**沒有東西可壓縮**(session 太小),會直接 return、**根本不觸發**
`session_before_compact`。此時 flag 會殘留;若之後你打內建 `/compact`,
handler 會誤以為是 `/kvc` 觸發而把它劫持走。

因此 flag 帶過期時間:

- **正常情況**:`ctx.compact()` → 事件觸發是毫秒級的,2 分鐘是極寬鬆的上限,
  正常永遠不會等到過期。
- **异常情况**(無東西可壓縮等):flag 2 分鐘後自動失效,一切回到內建行為,
  避免一個殘留 flag 在之後很久的 `/compact` 上造成誤劫持。
- **其他清除時機**:handler 成功消費後立即清掉;`onError` 時清掉;
  `session_start` / `session_shutdown` 時清掉。

換句話說:2 分鐘不是「2 分鐘內要用」,而是「萬一事件沒來,最壞情況
2 分鐘後自動回到內建行為」的安全上限。

### 自動回退到內建 `/compact` 的條件

- 沒有捕獲到任何請求(全新 session)
- 模型或 provider 已變更(捕獲的請求與目前模型不符)
- 目前模型不是 OpenAI 相容 API(例如 Anthropic Messages)
- 自訂摘要請求失敗(連線錯誤、HTTP 錯誤、空回應)
- 使用者按 Esc 中斷 → 取消本次壓縮

以上任一情況都不會弄壞 session:最壞就是回到內建 /compact 的慢速路徑。

### 與 pi-cache-guardian 的相容性

pi-cache-guardian 會逐 turn 改寫 system prompt(golden prompt 最佳化),
並在 payload 注入 `prompt_cache_key`。`/kvc` 捕獲的是**實際發出的 payload**
(含這些改動),所以完全相容;程式也刻意**不**用
`ctx.getSystemPrompt()` 去比對捕獲的 system prompt(兩者本來就可能不同),
避免在有 prompt 最佳化擴展的環境永遠誤判回退。

## 六、與後端(llama.cpp / LM Studio)的配合

- **預設行為已足夠**:`--cache-prompt` 預設開啟、`--cache-reuse 0` 預設、
  LM Studio 預設支援前綴重用並保留快取池。
- **保持模型 loaded**:KV 存在伺服器記憶體裡,別讓 session 閒置太久把快取清掉,
  否則又變全量重填(只是退回 /compact 的速度,不會更糟)。
- 長上下文本地推理建議的 llama-server 參數:
  - `-fa 1`:flash attention(Vulkan 可用)
  - `-ctk q8_0 -ctv q8_0`:KV 量化,省 VRAM、可留更大 context
  - `-c`:context 開夠大;`-ngl`:盡可能上 GPU
- 唯一新增的設定是 kvc 自己的開關 `kvc.autoCompact`(預設 true,見上一節),
  pi 內建的 compaction settings 不用動。

## 七、實測數據(LM Studio,ornith-1.5-35b-a3b,41k tokens 上下文)

| 請求類型 | 耗時 |
|---|---|
| 全新生成 prefill | 107.8s |
| **/kvc 風格(前綴命中 + 600 tokens 生成)** | **33.3s** |
| 內建 /compact 風格(全量重填) | 121.3s |

上下文越大,差距越大(200k+ 時內建要幾分鐘到十幾分鐘,
/kvc 只剩指令 prefill + 摘要生成時間)。

## 八、實作細節(hooks 與狀態)

- `before_provider_request`:捕獲最後一個 OpenAI chat-completions payload
  (要求 `messages[0]` 是 system、且不是內建摘要提示詞;
  模型切換時作廢捕獲)。
- `registerCommand("kvc")`:檢查前置條件 → 設定 armed 標誌 →
  `ctx.compact({ customInstructions, onComplete, onError })`。
- `agent_settled`(每次 agent run 完全 settled):讀 `kvc.autoCompact`
  (預設 true)→ 用 pi 自己的 `getContextUsage()` 算出百分比,
  ≥85% 且前置條件與 `/kvc` 相同(有捕獲、OpenAI 相容、模型一致,不用 force)
  → arm 後呼叫同一個 `ctx.compact()`,所以回退邏輯完全共用。
- `session_before_compact`:若 flag 有效(manual 觸發),**或**這是自動壓縮
  (threshold/overflow)且 `kvc.autoCompact` 開啟 →
  以「捕獲 payload + 追加指令」直接呼叫 `<baseUrl>/chat/completions`
  (`stream:false`;一次瞬時錯誤重試;模型若誤用 tools 則無 tools 重試一次)
  → 回傳自訂 `compaction`(摘要 + `firstKeptEntryId` + `tokensBefore` +
  pi 同款的 `<read-files>` / `<modified-files>` 檔案追蹤,讓跨次壓縮的
  檔案記錄能持續累積)。
- 模組級狀態(捕獲的 payload、armed 時間)在 `session_start` /
  `session_shutdown` 時清除。

## 九、檔案位置

| 檔案 | 說明 |
|---|---|
| `~/.pi/agent/extensions/kvc.ts` | 擴展本体 |
| `~/.pi/agent/extensions/kvc.md` | 英文說明 |
| `~/.pi/agent/extensions/kvc_zh.md` | 本檔(繁體中文說明) |

在 pi 中 `/reload` 或重新開啟後,`/kvc` 即可使用;`/compact` 保持原樣、隨時可用。
