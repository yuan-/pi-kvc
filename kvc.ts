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
 * Auto-compaction (default ON)
 * ----------------------------
 * Mirrors pi's own auto-compact design: after every agent run has fully
 * settled, the context usage is checked with pi's own accounting
 * (ctx.getContextUsage() - the same numbers shown in the footer) and a kvc
 * compaction is armed once it reaches 85% of the model window. It also takes
 * over pi's built-in threshold/overflow auto-compactions, so on small windows
 * (where pi's `contextWindow - reserveTokens` fires before 85%) automatic
 * compactions still go through the KV-cache path.
 *
 * Toggle: { "kvc": { "autoCompact": false } } in <project>/.pi/settings.json
 * or ~/.pi/agent/settings.json. Default is true (on).
 *
 * Safety
 * ------
 * /kvc only takes over a compaction triggered by /kvc itself, or an automatic
 * compaction while kvc.autoCompact is on (armed flag with a 2-minute expiry).
 * The manual built-in /compact is untouched. Everything falls back to the
 * built-in compaction automatically when:
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
  AgentSettledEvent,
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
/** Set by `/kvc force`; skips the captured/current model-id check. */
let compatForce = false;
dbg(`module loaded (pid=${process.pid})`);

const SUMMARIZER_SYSTEM_PREFIX = "You are a context summarization assistant";
/**
 * Auto-compaction trigger point: percentage of the model's context window.
 * Mirrors pi's own auto-compact formula (contextTokens > contextWindow -
 * reserveTokens) with an implicit reserve of 15% of the window. Checked after
 * every settled agent run, using pi's own context accounting.
 */
const AUTO_COMPACT_THRESHOLD_PERCENT = 85;
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

/** Read kvc.autoCompact from project settings, then global settings. Default: on. */
function readAutoCompactEnabled(cwd: string): boolean {
  const candidates = [join(cwd, ".pi", "settings.json"), join(homedir(), ".pi", "agent", "settings.json")];
  for (const p of candidates) {
    try {
      const s = JSON.parse(readFileSync(p, "utf8"));
      const v = s?.kvc?.autoCompact;
      if (typeof v === "boolean") return v;
    } catch {
      /* missing/invalid file - try next */
    }
  }
  return true; // default on
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

/**
 * Model-id comparison tolerant of pi version differences: some pi versions
 * prefix the provider into the payload `model` field ("lmstudio/foo" vs
 * "foo"). Only a leading "<currentProvider>/" segment is tolerated, so ids
 * that legitimately contain slashes ("qwen/qwen3.6-35b-a3b") are unaffected.
 */
function modelIdMatches(capturedId: string, currentId: string, provider?: string): boolean {
  if (capturedId === currentId) return true;
  if (provider && capturedId === `${provider}/${currentId}`) return true;
  return false;
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
  // Some models (e.g. qwen3.x via LM Studio) carry the system prompt in
  // OpenAI's newer "developer" role - treat both as the system prompt.
  if (!sysMsg || (sysMsg.role !== "system" && sysMsg.role !== "developer") || typeof sysMsg.content !== "string") {
    throw new Error("captured payload has no system/developer message");
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
    // Accept both "system" and OpenAI's newer "developer" role (used by e.g.
    // qwen3.x on LM Studio) as the system prompt - otherwise every request
    // from those models would be skipped and nothing would ever be captured.
    const sys = p.messages[0];
    if (!sys || typeof sys !== "object" || (sys.role !== "system" && sys.role !== "developer") || typeof sys.content !== "string") {
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
    const newId = event.model?.id ? String(event.model.id) : "";
    const newProvider = (event.model as any)?.provider != null ? String((event.model as any).provider) : undefined;
    if (captured && !modelIdMatches(captured.modelId, newId, newProvider)) {
      dbg(`model_select ${event.model?.id} != captured ${captured.modelId} -> invalidate`);
      captured = null;
    }
  });
  pi.on("session_start", (e: any) => {
    dbg(`session_start reason=${e?.reason} (clear captured/armed)`);
    captured = null;
    compatArmedUntil = 0;
    compatForce = false;
  });
  pi.on("session_shutdown", (e: any) => {
    dbg(`session_shutdown reason=${e?.reason}`);
    captured = null;
    compatArmedUntil = 0;
    compatForce = false;
  });

  // 3) Take over compactions triggered by /kvc, plus automatic ones while
  //    kvc.autoCompact is on. A plain manual /compact always stays built-in.
  pi.on("session_before_compact", async (event, ctx) => {
    const armed = Date.now() <= compatArmedUntil;
    const autoTakeover = event.reason !== "manual" && readAutoCompactEnabled(ctx.cwd);
    dbg(`session_before_compact reason=${event.reason} armed=${armed} autoTakeover=${autoTakeover} captured=${Boolean(captured)} force=${compatForce}`);
    if (!armed && !autoTakeover) return; // not ours: manual /compact, or auto-compaction with kvc.autoCompact off -> untouched
    compatArmedUntil = 0;
    const force = compatForce;
    compatForce = false;

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
    if (!captured) {
      notify("kvc: no cached request for the current model - using built-in compaction", "warning");
      return;
    }
    if (!modelIdMatches(captured.modelId, model.id, String(model.provider))) {
      if (!force) {
        notify(`kvc: model mismatch (captured="${captured.modelId}" current="${model.id}") - using built-in compaction`, "warning");
        return;
      }
      notify(`kvc: forced - captured model "${captured.modelId}" != current "${model.id}"; prefix cache may miss`, "info");
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
      "KV-cache-compatible compaction (like /compact, but the summary request reuses the server's KV cache - no full re-prefill on llama.cpp/LM Studio backends). Optional: /kvc <focus instructions>; /kvc force skips the model-id check",
    handler: async (args, ctx) => {
      let instructions = (args ?? "").trim();
      let force = false;
      const forceMatch = instructions.match(/^--?force\b/i);
      if (forceMatch) {
        force = true;
        instructions = instructions.slice(forceMatch[0].length).trim();
      }
      const focus = instructions || undefined;
      dbg(`/kvc invoked force=${force} args=${JSON.stringify(args ?? "")} captured=${Boolean(captured)} capturedModel=${captured?.modelId ?? "-"} ctxModel=${ctx.model ? `${ctx.model.provider}/${ctx.model.id} api=${ctx.model.api}` : "-"}`);
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
      if (!modelIdMatches(captured.modelId, ctx.model.id, String(ctx.model.provider)) && !force) {
        ctx.ui.notify(`kvc: model mismatch (captured="${captured.modelId}" current="${ctx.model.id}") - run one turn with the current model first, or /kvc force, or /compact`, "warning");
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
      compatForce = force;
      ctx.compact({
        customInstructions: focus,
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

  // 5) Auto-compaction at AUTO_COMPACT_THRESHOLD_PERCENT of the context window.
  //    Mirrors pi's own auto-compact checkpoints: checked after every agent run
  //    has fully settled (retries and queued continuations drained), using pi's
  //    own context accounting. The trigger only ARMS a compaction via the same
  //    ctx.compact() path as /kvc, so all fallbacks apply unchanged.
  function maybeAutoCompact(ctx: ExtensionContext): void {
    try {
      if (!readAutoCompactEnabled(ctx.cwd)) return;
      if (Date.now() <= compatArmedUntil) return; // compaction pending/running; the armed TTL also cools down after failures

      const model = ctx.model;
      if (!model || typeof model.contextWindow !== "number" || model.contextWindow <= 0) return;

      let percent: number | null = null;
      try {
        // pi's own accounting (last assistant usage / estimate), same as the footer display.
        const usage = ctx.getContextUsage();
        if (usage && typeof usage.percent === "number") percent = usage.percent;
      } catch {
        /* unknown - skip */
      }
      if (percent == null || !Number.isFinite(percent)) return; // context size unknown until the next LLM response

      if (percent < AUTO_COMPACT_THRESHOLD_PERCENT) {
        dbg(`auto-compact check: ${percent.toFixed(1)}% < ${AUTO_COMPACT_THRESHOLD_PERCENT}% -> skip`);
        return;
      }

      // Same preconditions as /kvc (without force): on mismatch, fall through to pi's own compaction.
      if (!captured) {
        dbg(`auto-compact SKIP: context at ${percent.toFixed(1)}% but no captured request`);
        return;
      }
      if (model.api !== "openai-completions") {
        dbg(`auto-compact SKIP: api=${model.api} is not OpenAI-compatible`);
        return;
      }
      if (!modelIdMatches(captured.modelId, model.id, String(model.provider))) {
        dbg(`auto-compact SKIP: model mismatch (captured="${captured.modelId}" current="${model.id}")`);
        return;
      }

      compatArmedUntil = Date.now() + ARMS_TTL_MS;
      compatForce = false;
      const pct = Math.round(percent);
      dbg(`auto-compact TRIGGER at ${pct}% of ${model.contextWindow} tokens (threshold=${AUTO_COMPACT_THRESHOLD_PERCENT}%)`);

      try {
        if (ctx.hasUI) ctx.ui.notify(`kvc: context at ${pct}% of window - auto-compacting via KV cache`, "info");
        else console.log(`[kvc] context at ${pct}% of window - auto-compacting`);
      } catch {
        /* non-fatal */
      }

      ctx.compact({
        onComplete: (result) => {
          try {
            const msg = `kvc auto-compact done: ${result.tokensBefore} -> ~${result.estimatedTokensAfter ?? "?"} context tokens`;
            if (ctx.hasUI) ctx.ui.notify(msg, "info");
            else console.log(`[kvc] ${msg}`);
          } catch {
            /* non-fatal */
          }
        },
        onError: (err) => {
          compatArmedUntil = 0; // allow a retry on the next settled run
          try {
            if (ctx.hasUI) ctx.ui.notify(`kvc auto-compact failed: ${err.message}`, "error");
            else console.error(`[kvc] auto-compact failed: ${err.message}`);
          } catch {
            /* non-fatal */
          }
        },
      });
    } catch (err: any) {
      // e.g. ctx went stale during a session replacement while this ran
      dbg(`auto-compact check error: ${String(err?.message ?? err)}`);
    }
  }

  pi.on("agent_settled", (_event: AgentSettledEvent, ctx) => {
    void maybeAutoCompact(ctx); // do not block the settle/idle path on compaction
  });
}
