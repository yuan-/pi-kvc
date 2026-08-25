# /kvc — KV-cache-compatible compaction

`kvc.ts` 为 pi 增加 `/kvc` 指令(不改动内建 `/compact`)。

## 解决的问题

内建 `/compact` 的 summarization 请求使用**不同的 system prompt** + 重新序列化的对话,
prompt 前缀与之前任何请求都不匹配 → llama.cpp/LM Studio 后端必须把**整个对话重新 prefill**
(实测:41k tokens 约 2 分钟;上下文越大越痛)。

## /kvc 的做法

把 summarization 请求构造成**上一个正常请求的前缀扩展**:

```
[captured payload: system + 完整对话 + tools,与上次发送逐字节相同]
  + [一条追加的 user message:summarization 指令]
```

服务器对前缀命中 KV cache,只需 prefill 追加的指令(约 1k tokens)。
压缩后的下一次正常请求也只 prefill 小的 summary + kept messages。

指令中明确告诉模型:顶部的 system prompt / tools / guidelines 是常驻配置,
**不要写进 summary** → 压缩后的上下文(system prompt + summary)中 system prompt 只出现一次,
不存在"放二次"的问题。若模型仍逐字复制了 system prompt,后处理会裁掉开头重复部分。

## 用法

- `/kvc` — 压缩(行为等同 /compact,但走 KV cache)
- `/kvc <focus>` — 带重点提示,例如 `/kvc 保留最近测试失败的信息`

前提:当前 session 至少完成过一个 agent turn(需要已捕获的请求作为前缀);
模型/provider 变更、请求失败等情况会自动回退到内建 /compact 逻辑。

## 后端要求(已实测通过:LM Studio + llama.cpp/Vulkan)

服务器必须支持跨请求的 KV 前缀重用:
- llama-server:`--cache-prompt`(默认开启)、`--cache-reuse 0`(默认)
- LM Studio:默认行为即支持,且会保留多个已缓存前缀(池)
- 保持模型在服务器中保持加载(keep-alive),不要让 session 闲置太久把 KV 清掉

参考启动参数(llama-server,长上下文本地场景):
`-fa 1`(flash attention, Vulkan 可用)、`-ctk q8_0 -ctv q8_0`(量化 KV,省 VRAM)、
`-c` 设足够大的 context、`-ngl` 尽量上 GPU。

## 实现要点

- `before_provider_request`:捕获最后一个 OpenAI chat-completions payload
  (跳过内建 summarization 请求)。
- `/kvc`:armed 标志(2 分钟有效)→ `ctx.compact()` → `session_before_compact`
  hook 内用 captured payload + 追加指令直接请求 `<baseUrl>/chat/completions`
  (`stream:false`),返回自定义 `compaction` 条目(含 pi 同款
  `<read-files>/<modified-files>` 文件追踪)。
- 与 pi-cache-guardian 兼容:golden prompt 改写、`prompt_cache_key` 注入均不影响
  (capture 到的是实际发出的 payload)。

## 实测(LM Studio, ornith-1.5-35b-a3b, 41k tokens 上下文)

| 请求 | 耗时 |
|---|---|
| 全新 prefill | 107.8s |
| /kvc 风格(前缀 + 指令,含 600 token 生成) | 33.3s |
| 内建 /compact 风格(全量重填) | 121.3s |
