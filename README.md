# pi-kvc

KV-cache-compatible context compaction for [pi](https://github.com/badlogic/pi-mono) (`pi-coding-agent`).

Adds a `/kvc` command: like the built-in `/compact`, but the summarization
request is sent as a **prefix extension of the last normal agent request**,
so llama.cpp-based local servers (LM Studio, `llama-server`) reuse their KV
cache instead of re-prefilling the entire conversation.

## Why

The built-in `/compact` sends a summarization request with a *different*
system prompt and a re-serialized conversation. On a local llama.cpp server
the prompt prefix no longer matches any cached KV, so the whole conversation
is re-prefilled (measured: ~2 minutes for 41k tokens, and it grows with
context size). `/kvc` avoids that:

| Request style (41k-token context, local 35B MoE) | Time |
|---|---|
| Full re-prefill (built-in `/compact` style) | 121.3 s |
| `/kvc` style (cached prefix + ~1k appended instruction) | 33.3 s |

## How it works

1. `before_provider_request` captures the last normal OpenAI
   chat-completions payload (system + conversation + tools, exactly as sent
   to the server).
2. `/kvc` arms a 2-minute flag and calls pi's `ctx.compact()`.
3. In `session_before_compact`, the extension POSTs the captured payload
   **byte-identical** plus one appended user message containing the
   summarization instruction to `<baseUrl>/chat/completions`. Only the
   appended instruction is prefilled; the whole prefix is a KV cache hit.
4. The instruction tells the model that the system prompt / tools /
   guidelines at the top of the context are permanent configuration that
   must NOT be restated — the post-compaction context therefore contains
   the system prompt exactly once (a post-processing guard also strips a
   verbatim leading copy if the model repeats it anyway).
5. On any failure (no capture yet, model changed, non-OpenAI provider,
   request error) `/kvc` automatically falls back to the built-in
   compaction.
6. **Auto-compaction (on by default)**: after every agent run has fully
   settled — pi's own checkpoint for its built-in auto-compact — the context
   usage is checked with pi's own accounting (`getContextUsage()`, the same
   number shown in the footer). At **85% of the model window** a kvc
   compaction is armed and runs through the exact same path as `/kvc`.
   While enabled, automatic (threshold/overflow) compactions are also taken
   over, so on small windows where pi's `contextWindow - reserveTokens`
   threshold fires before 85%, they still use the KV-cache path. Manual
   `/compact` always stays built-in.

The summary format matches pi's built-in one (`## Goal` / `### Done` /
`### In Progress` / ...), including `<read-files>` / `<modified-files>`
tracking, so file tracking keeps working across compactions.

## Requirements

- `pi-coding-agent` (pi)
- An OpenAI-compatible chat-completions server that reuses KV prefixes
  across requests — LM Studio or `llama-server` with default settings both
  work (`--cache-prompt` / `--cache-reuse 0` are on by default)
- The model loaded and kept alive on that server (don't let the server
  unload it / clear the KV pool between the normal turn and `/kvc`)
- At least one completed agent turn in the session (the capture)

## Install

```bash
git clone git@github.com:yuan-/pi-kvc.git
cp pi-kvc/kvc.ts ~/.pi/agent/extensions/kvc.ts
```

(Windows: copy to `%USERPROFILE%\.pi\agent\extensions\kvc.ts`.)

Then in pi:

1. `/reload` (or start a new session)
2. run one normal agent turn
3. `/kvc`

## Configuring the local server in pi

`~/.pi/agent/models.json` — example for LM Studio's local server:

```json
{
  "providers": {
    "lmstudio": {
      "baseUrl": "http://localhost:1234/v1",
      "api": "openai-completions",
      "apiKey": "lm-studio",
      "models": [
        { "id": "<your-model-name>", "contextWindow": 131072, "maxTokens": 16384 }
      ]
    }
  }
}
```

## Usage

- `/kvc` — compact (same semantics as `/compact`)
- `/kvc <focus>` — compact with extra instructions, e.g.
  `/kvc keep the last failing test`
- `/kvc force` — skip the captured/current model-id check (see below). Can be
  combined: `/kvc force keep the test output`

## Auto-compaction (on by default)

So you don't have to remember `/kvc` on long local-model sessions: when the
context reaches **85% of the window**, kvc arms itself and runs the same
KV-cache-compatible compaction automatically — no full re-prefill.

It mirrors pi's own auto-compact design:

- checked after every agent run has fully settled (retries and queued
  continuations drained), using pi's own context accounting — the same
  percentage shown in the footer
- also takes over pi's built-in threshold/overflow auto-compactions, so even
  on small windows (where `contextWindow - reserveTokens` fires before 85%)
  automatic compactions go through the KV-cache path
- all `/kvc` fallbacks apply: no capture yet / model changed / non-OpenAI
  provider → built-in compaction runs instead; manual `/compact` always stays
  on the built-in path (use `/kvc` for the fast one)

Toggle in `~/.pi/agent/settings.json` or `<project>/.pi/settings.json`
(project wins), default is **on**:

```json
{ "kvc": { "autoCompact": false } }
```

Auto-trigger decisions (`TRIGGER` / `SKIP` with the reason) are logged to
`kvc-debug.log` in your temp directory.

Notes:

- Run one normal turn first; until then the extension has nothing to
  capture and will tell you to use `/compact`.
- Switching models invalidates the capture (run another turn, then `/kvc`).
- `model mismatch` — the model id recorded in the captured request differs
  from the current model id. If the mismatch is only cosmetic (e.g. a pi
  version formats the id differently), `/kvc force` proceeds anyway — the
  summary is still correct, only the prefix-cache benefit may be lost.
- The summary generation itself is as slow as a normal completion of the
  same length on your local model — `/kvc` only removes the re-prefill.
- Diagnostics (counts and timing only, no message content) are appended to
  `kvc-debug.log` in your system temp directory.

## Troubleshooting

- `kvc: no request cached yet` — run one agent turn, then `/kvc`.
- `kvc: model mismatch (captured="X" current="Y")` — the two ids are shown
  in the message. If X and Y are the same model in different formats, use
  `/kvc force`. If they are genuinely different models, run one turn with
  the current model, then `/kvc`.
- `kvc: ... falling back to built-in compaction` — the fallback ran the
  normal `/compact` flow; check `kvc-debug.log` in your temp dir for the
  recorded error.

## License

MIT — see [LICENSE](LICENSE)
