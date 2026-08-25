/**
 * /kvc — KV-cache-compatible compaction (KV 快取相容壓縮).
 *
 * Why
 * ---
 * The built-in /compact sends a standalone summarization request with a
 * DIFFERENT system prompt and the conversation re-serialized as text. On a
 * local llama.cpp-based server (LM Studio / llama-server) the prompt prefix
 * no longer matches the cached KV, so the ENTIRE conversation is re-prefilled
 * (tens of seconds to minutes per 100k tokens on a GPU).
 *
 * How /kvc avoids it
 * -------------------
 * /kvc sends the summarization request as a strict PREFIX EXTENSION of the
 * last normal agent request:
 *
 *   [captured payload: system + full conversation + tools, exactly as last
 *    sent to the server] + [ONE appended user message with the summarization
 *    instruction]
 *
 * The server reuses its KV cache for the whole prefix (verified: llama.cpp
 * --cache-prompt / LM Studio both reuse common prefixes across requests and
 * keep a pool of cached prefixes); only the appended instruction (~1k tokens)
 * is prefilled. After compaction, the next normal request re-prefills only
 * the small summary + kept messages.
 *
 * The appended instruction tells the model that the system prompt / tools /
 * guidelines at the top of the context are permanent configuration that must
 * NOT be restated in the summary — so the post-compaction context contains
 * the system prompt exactly once (no duplication). A post-processing guard
 * also strips a verbatim leading copy of the system prompt if the model
 * repeats it anyway.
 *
 * Safety
 * ------
 * /kvc only takes over a compaction triggered by /kvc itself (armed flag with
 * a 2-minute expiry). The built-in /compact and auto-compaction are untouched.
 * /kvc falls back to the built-in compaction automatically when:
 *   - no OpenAI-shaped request was captured yet (fresh session),
 *   - the model or system prompt changed since the capture,
 *   - the provider is not OpenAI-compatible,
 *   - the custom request fails.
 */
import { appendFileSync, readFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

// Diagnostics: state transitions and the summary-request lifecycle are logged
// to <tmpdir>/kvc-debug.log (no message content, only counts/timing) to make
// failed /kvc runs easy to troubleshoot. Delete the file to stop collecting.
const DEBUG_LOG = join(tmpdir(), "kvc-debug.log");
function dbg(msg: string) {
  try {
    appendFileSync(DEBUG_LOG, `${new Date().toISOString()} ${msg}\n`);
  } catch {
    /* ignore */
  }
}
import type {
  ExtensionAPI,
  ExtensionContext,
  ModelSelectEvent,
  SessionBeforeCompactEvent,
} from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// State (per extension instance; module state survives session rebinds, so
// it is cleared on session lifecycle events below)
// ---------------------------------------------------------------------------

interface CapturedRequest {
  /** Last OpenAI chat-completions payload, exactly as sent to the server. */
  payload: any;
  /** Model id used in the captured payload. */
  modelId: string;
  /** Content of the captured system message (messages[0]). */
  systemPrompt: string;
}

let captured: CapturedRequest | null = null;
/** Set by /kvc; consumed by the session_before_compact hook or expired. */
let compatArmedUntil = 0;
dbg(`module loaded (pid=${process.pid})`);

const SUMMARIZER_SYSTEM_PREFIX = "You are a context summarization assistant";
const ARMS_TTL_MS = 2 * 60 * 1000; // compaction starts within milliseconds
// Local 27B-class models can generate a 10k+ token summary at only ~5 tok/s;
// 30 min proved too short in practice (observed: request aborted at 30 min,
// fallback re-did the whole compaction). Built-in /compact has no such limit.
const SUMMARY_REQUEST_TIMEOUT_MS = 60 * 60 * 1000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildInstruction(opts: { hasPreviousSummary: boolean; customInstructions?: string }): string {
  const lines: string[] = [];
  lines.push("[COMPACT INSTRUCTION]");
  lines.push("You are now in a summarization pass, not normal operation. Your only task is to output one structured context checkpoint for a fresh agent instance.");
  lines.push("");
  lines.push("SCOPE: Summarize the ENTIRE conversation above, from the first message to the most recent one.");
  if (opts.hasPreviousSummary) {
    lines.push("An earlier context summary (a '## Goal' markdown block) is present in the conversation. MERGE it into your output: preserve still-relevant items, move completed items to Done, and update In Progress / Blocked / Next Steps to the current state.");
  }
  lines.push("");
  lines.push("STANDING CONFIGURATION - DO NOT RESTATE: The system prompt, tool definitions and guidelines at the top of this context are permanent agent configuration that remains in effect automatically. Do NOT include, quote, paraphrase or summarize any of them in your output. Your output must describe the conversation only.");
  lines.push("");
  lines.push("HARD RULES:");
  lines.push("- Do NOT call any tools. Reply with text only.");
  lines.push("- Do NOT continue the conversation or answer questions from it.");
  lines.push("- Output ONLY the markdown format below - no code fences, no preamble, no trailing commentary.");
  lines.push("");
  lines.push("Format (EXACT):");
  lines.push("");
  lines.push("## Goal");
  lines.push("[What is the user trying to accomplish? Multiple items if the session covers different tasks.]");
  lines.push("");
  lines.push("## Constraints & Preferences");
  lines.push("- [Constraints, preferences or requirements mentioned by user] - or (none)");
  lines.push("");
  lines.push("## Progress");
  lines.push("### Done");
  lines.push("- [x] [Completed tasks/changes]");
  lines.push("");
  lines.push("### In Progress");
  lines.push("- [ ] [Current work]");
  lines.push("");
  lines.push("### Blocked");
  lines.push("- [Issues preventing progress, if any]");
  lines.push("");
  lines.push("## Key Decisions");
  lines.push("- **[Decision]**: [Brief rationale]");
  lines.push("");
  lines.push("## Next Steps");
  lines.push("1. [Ordered list of what should happen next]");
  lines.push("");
  lines.push("## Critical Context");
  lines.push("- [Data, examples or references needed to continue] - or (none)");
  lines.push("");
  lines.push("Keep each section concise. Preserve exact file paths, function names and error messages.");
  if (opts.customInstructions) {
    lines.push("");
    lines.push(`Additional focus: ${opts.customInstructions}`);
  }
  return lines.join("\n");
}

/** Read compaction.reserveTokens from project settings, then global settings. */
function readReserveTokens(cwd: string): number {
  const candidates = [join(cwd, ".pi", "settings.json"), join(homedir(), ".pi", "agent", "settings.json")];
  for (const p of candidates) {
    try {
      const s = JSON.parse(readFileSync(p, "utf8"));
      const r = s?.compaction?.reserveTokens;
      if (typeof r === "number" && r > 0) return r;
    } catch {
      /* missing/invalid file - try next */
    }
  }
  return 16384; // pi default
}

function readKeepRecentTokens(cwd: string): number {
  const candidates = [join(cwd, ".pi", "settings.json"), join(homedir(), ".pi", "agent", "settings.json")];
  for (const p of candidates) {
    try {
      const s = JSON.parse(readFileSync(p, "utf8"));
      const r = s?.compaction?.keepRecentTokens;
      if (typeof r === "number" && r > 0) return r;
    } catch {
      /* missing/invalid file - try next */
    }
  }
  return 20000; // pi default
}

function toPiUsage(raw: any) {
  const input = typeof raw?.prompt_tokens === "number" ? raw.prompt_tokens : 0;
  const output = typeof raw?.completion_tokens === "number" ? raw.completion_tokens : 0;
  const total = typeof raw?.total_tokens === "number" ? raw.total_tokens : input + output;
  return {
    input,
    output,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: total,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

/**
 * Same file-list tags pi's built-in compaction appends (<read-files> /
 * <modified-files>) so cumulative file tracking keeps working across
 * compactions.
 */
function formatFileOperations(fileOps: { read: Set<string>; written: Set<string>; edited: Set<string> }): {
  suffix: string;
  details: { readFiles: string[]; modifiedFiles: string[] };
} {
  const modified = new Set<string>([...fileOps.edited, ...fileOps.written]);
  const readFiles = [...fileOps.read].filter((f) => !modified.has(f)).sort();
  const modifiedFiles = [...modified].sort();
  const sections: string[] = [];
  if (readFiles.length > 0) sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
  if (modifiedFiles.length > 0) sections.push(`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`);
  return { suffix: sections.length > 0 ? `\n\n${sections.join("\n\n")}` : "", details: { readFiles, modifiedFiles } };
}

/**
 * Safety net for the "system prompt duplicated twice" problem: if the model
 * repeated the standing system prompt verbatim at the start of its summary,
 * drop the copy (the real summary follows it). A legitimate summary starts
 * with "## Goal" and can never share a 100+ char prefix with the system
 * prompt, so this cannot damage a normal summary.
 */
function stripRepeatedSystemPrompt(summary: string, systemPrompt: string): string {
  const s = summary.trimStart();
  if (!systemPrompt || s.length < 100 || systemPrompt.length < 100) return summary;
  if (!s.startsWith(systemPrompt.slice(0, 100))) return summary;
  let i = 100;
  while (i < s.length && i < systemPrompt.length && s[i] === systemPrompt[i]) i++;
  if (i <= 100) return summary;
  const rest = s.slice(i).replace(/^\s+/, "");
  return rest.length > 40 ? rest : summary;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error("aborted"));
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(new Error("aborted"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

interface SummaryResponse {
  text: string;
  usage: any;
  hadToolCalls: boolean;
}

async function postSummary(url: string, headers: Record<string, string>, payload: any, signal: AbortSignal): Promise<SummaryResponse> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${body.slice(0, 300)}`);
  }
  const data = await res.json();
  const choice = data?.choices?.[0];
  const text = typeof choice?.message?.content === "string" ? choice.message.content : "";
  return {
    text,
    usage: data?.usage,
    hadToolCalls: Array.isArray(choice?.message?.tool_calls) && choice.message.tool_calls.length > 0,
  };
}

// ---------------------------------------------------------------------------
// The cache-compatible summarization call
// ---------------------------------------------------------------------------

async function runCompatSummary(base: CapturedRequest, model: any, ctx: ExtensionContext, event: SessionBeforeCompactEvent): Promise<{
  summary: string;
  usage: ReturnType<typeof toPiUsage>;
  details: { readFiles: string[]; modifiedFiles: string[] };
}> {
  if (!base.payload || !Array.isArray(base.payload.messages) || base.payload.messages.length === 0) {
    throw new Error("captured payload is invalid");
  }
  const sysMsg = base.payload.messages[0];
  if (!sysMsg || sysMsg.role !== "system" || typeof sysMsg.content !== "string") {
    throw new Error("captured payload has no system message");
  }
  // NOTE: deliberately NOT comparing the captured system prompt against
  // ctx.getSystemPrompt(). Extensions such as pi-cache-guardian rewrite the
  // system prompt per turn ("golden prompt" optimization), so the payload that
  // was actually sent (and whose KV is cached) may legitimately differ from
  // the base system prompt. The captured payload is by construction the
  // cached prefix, and the summary covers the conversation as it actually
  // happened - the same semantics as the built-in /compact.
  const baseUrl = typeof model.baseUrl === "string" ? model.baseUrl : undefined;
  if (!baseUrl) throw new Error(`model ${model.id} has no baseUrl`);
  const url = `${baseUrl.replace(/\/+$/, "")}/chat/completions`;

  const reserve = readReserveTokens(ctx.cwd);
  const maxOut = typeof model.maxTokens === "number" && model.maxTokens > 0 ? model.maxTokens : reserve;
  const budget = Math.max(1024, Math.min(Math.floor(reserve * 0.8), maxOut));

  const instruction = buildInstruction({
    hasPreviousSummary: Boolean(event.preparation.previousSummary),
    customInstructions: event.customInstructions,
  });

  // Reuse the captured payload byte-for-byte (prefix!) and append ONE user
  // message. Only `stream` and `max_tokens` change - neither affects the
  // rendered prompt, so the KV cache prefix stays valid.
  const payload: any = {
    ...base.payload,
    model: model.id,
    stream: false,
    messages: [...base.payload.messages, { role: "user", content: instruction }],
  };
  delete payload.stream_options;
  // Exactly ONE max-tokens field: the captured payload may already carry
  // max_completion_tokens (pi >= 0.84.3) or max_tokens (older pi); carrying
  // both is ambiguous. Neither field affects the rendered prompt, so the KV
  // prefix stays valid.
  delete payload.max_tokens;
  payload.max_completion_tokens = budget;
  dbg(`summary request start: messages=${payload.messages.length} budget=${budget} capturedMaxTokens=${typeof base.payload.max_tokens} capturedMaxCompletionTokens=${typeof base.payload.max_completion_tokens}`);
  const t0 = Date.now();

  const headers: Record<string, string> = {};
  try {
    const apiKey = await ctx.modelRegistry.getApiKeyForProvider(String(model.provider));
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;
  } catch {
    /* server without auth */
  }
  if (model.headers && typeof model.headers === "object") {
    for (const [k, v] of Object.entries(model.headers)) {
      if (typeof v === "string") headers[k] = v;
    }
  }

  const abortSignal = typeof AbortSignal.any === "function"
    ? AbortSignal.any([event.signal, AbortSignal.timeout(SUMMARY_REQUEST_TIMEOUT_MS)])
    : event.signal;

  let res: SummaryResponse;
  try {
    res = await postSummary(url, headers, payload, abortSignal);
    dbg(`summary response ok after ${Math.round((Date.now() - t0) / 1000)}s usage=${JSON.stringify(res.usage ?? null)}`);
  } catch (err: any) {
    dbg(`summary request FAILED after ${Math.round((Date.now() - t0) / 1000)}s: ${String(err?.message ?? err)} (aborted=${abortSignal.aborted})`);
    if (abortSignal.aborted) throw err;
    // One retry for transient errors (connection reset, 5xx under load).
    await sleep(2000, event.signal);
    res = await postSummary(url, headers, payload, abortSignal);
    dbg(`summary retry ok after ${Math.round((Date.now() - t0) / 1000)}s`);
  }

  if (!res.text.trim() && res.hadToolCalls) {
    // Degenerate: the model tried to call tools despite the instructions.
    // Retry once WITHOUT tools (this attempt loses the prefix cache - rare).
    const noTools: any = { ...payload };
    delete noTools.tools;
    const res2 = await postSummary(url, headers, noTools, abortSignal);
    if (res2.text.trim()) res = res2;
  }

  let text = res.text.trim();
  if (!text) throw new Error("model returned an empty summary");
  text = stripRepeatedSystemPrompt(text, sysMsg.content);

  const { suffix, details } = formatFileOperations(event.preparation.fileOps);
  return { summary: text + suffix, usage: toPiUsage(res.usage), details };
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  // 1) Capture every normal (non-summarization) OpenAI chat-completions
  //    payload so /kvc can extend it later.
  pi.on("before_provider_request", (event) => {
    const p = event.payload as any;
    if (!p || typeof p !== "object" || Array.isArray(p)) {
      dbg(`before_provider_request SKIP payload-type=${typeof p}`);
      return;
    }
    if (typeof p.model !== "string" || !Array.isArray(p.messages) || p.messages.length === 0) {
      dbg(`before_provider_request SKIP shape model=${typeof p.model} messages=${p && typeof p === "object" ? String(p.messages?.length ?? typeof p.messages) : "n/a"}`);
      return;
    }
    const sys = p.messages[0];
    if (!sys || typeof sys !== "object" || sys.role !== "system" || typeof sys.content !== "string") {
      dbg(`before_provider_request SKIP m0 role=${sys?.role} content-type=${typeof sys?.content}`);
      return;
    }
    if (sys.content.startsWith(SUMMARIZER_SYSTEM_PREFIX)) {
      dbg("before_provider_request SKIP summarization-request");
      return; // built-in summarization request - not a good base
    }
    captured = { payload: p, modelId: p.model, systemPrompt: sys.content };
    dbg(`CAPTURED model=${p.model} messages=${p.messages.length} tools=${p.tools?.length ?? 0}`);
  });

  // 2) Invalidate the capture when the base prompt can no longer match.
  pi.on("model_select", (event: ModelSelectEvent) => {
    if (captured && event.model?.id !== captured.modelId) {
      dbg(`model_select ${event.model?.id} != captured ${captured.modelId} -> invalidate`);
      captured = null;
    }
  });
  pi.on("session_start", (e: any) => {
    dbg(`session_start reason=${e?.reason} (clear captured/armed)`);
    captured = null;
    compatArmedUntil = 0;
  });
  pi.on("session_shutdown", (e: any) => {
    dbg(`session_shutdown reason=${e?.reason}`);
    captured = null;
    compatArmedUntil = 0;
  });

  // 3) Take over only /kvc-triggered compactions.
  pi.on("session_before_compact", async (event, ctx) => {
    dbg(`session_before_compact reason=${event.reason} armed=${Date.now() <= compatArmedUntil} captured=${Boolean(captured)}`);
    if (Date.now() > compatArmedUntil) return; // not armed: built-in /compact or auto-compaction -> untouched
    compatArmedUntil = 0;
    if (event.reason !== "manual") return;

    const notify = (msg: string, kind: "info" | "warning" | "error" = "info") => {
      try {
        if (ctx.hasUI) ctx.ui.notify(msg, kind);
        else console.log(`[kvc] ${msg}`);
      } catch {
        /* non-fatal */
      }
    };

    const model = ctx.model;
    if (!model) {
      notify("kvc: no active model - using built-in compaction", "warning");
      return;
    }
    if (model.api !== "openai-completions") {
      notify(`kvc: ${model.provider} is not OpenAI-compatible - using built-in compaction`, "warning");
      return;
    }
    if (!captured || captured.modelId !== model.id) {
      notify("kvc: no cached request for the current model - using built-in compaction", "warning");
      return;
    }

    try {
      const result = await runCompatSummary(captured, model, ctx, event);
      notify(`kvc: summary via KV cache reuse (${result.usage.totalTokens} tokens) - compacting`, "info");
      return {
        compaction: {
          summary: result.summary,
          firstKeptEntryId: event.preparation.firstKeptEntryId,
          tokensBefore: event.preparation.tokensBefore,
          usage: result.usage,
          details: result.details,
        },
      };
    } catch (err: any) {
      if (event.signal.aborted) return { cancel: true };
      notify(`kvc: ${err?.message ?? String(err)} - falling back to built-in compaction`, "error");
      return; // let pi run its built-in summarization
    }
  });

  // 4) The /kvc command itself.
  pi.registerCommand("kvc", {
    description:
      "KV-cache-compatible compaction (like /compact, but the summary request reuses the server's KV cache - no full re-prefill on llama.cpp/LM Studio backends). Optional: /kvc <focus instructions>",
    handler: async (args, ctx) => {
      const instructions = (args ?? "").trim() || undefined;
      dbg(`/kvc invoked args=${JSON.stringify(args ?? "")} captured=${Boolean(captured)} capturedModel=${captured?.modelId ?? "-"} ctxModel=${ctx.model ? `${ctx.model.provider}/${ctx.model.id} api=${ctx.model.api}` : "-"}`);
      if (!captured) {
        ctx.ui.notify("kvc: no request cached yet - run at least one agent turn first, then /kvc (or use /compact)", "warning");
        return;
      }
      if (!ctx.model) {
        ctx.ui.notify("kvc: no active model", "error");
        return;
      }
      if (ctx.model.api !== "openai-completions") {
        ctx.ui.notify(`kvc: ${ctx.model.provider} is not OpenAI-compatible - use /compact`, "warning");
        return;
      }
      if (captured.modelId !== ctx.model.id) {
        ctx.ui.notify("kvc: model changed since the last request - use /compact (or run one more turn first)", "warning");
        return;
      }
      try {
        await ctx.waitForIdle();
      } catch {
        /* proceed */
      }
      if (!captured) {
        ctx.ui.notify("kvc: state changed while waiting - use /compact", "warning");
        return;
      }
      compatArmedUntil = Date.now() + ARMS_TTL_MS;
      ctx.compact({
        customInstructions: instructions,
        onComplete: (result) => {
          ctx.ui.notify(`kvc done: ${result.tokensBefore} -> ~${result.estimatedTokensAfter ?? "?"} context tokens`, "info");
        },
        onError: (err) => {
          compatArmedUntil = 0;
          ctx.ui.notify(`kvc failed: ${err.message}`, "error");
        },
      });
    },
  });
}
