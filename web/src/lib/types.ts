/**
 * Wire types shared with the Gaia backend (src/gaia/app/routes.py).
 */

/** Routing tier chosen by the reasoning router (spec 03). */
export type Routing = string;

/** Safety verdict attached to every reply (spec 07). */
export type Verdict = string;

/** Per-message emotional reading of the user's input (spec 07 intake). */
export interface Emotions {
  happiness: number;
  sadness: number;
  fear: number;
  disgust: number;
  anger: number;
  surprise: number;
}

/** A single chat turn rendered in the UI. */
export interface ChatMessage {
  id: string;
  role: 'user' | 'gaia';
  text: string;
  ts: string;
  /** True while tokens are still streaming into this message. */
  streaming?: boolean;
  /** Metadata returned with the final Gaia reply. */
  meta?: ReplyMeta;
  /** Live process-log streamed during the turn (before {@link meta} arrives). */
  events?: TurnEvent[];
  /** Set when the turn failed. */
  error?: string;
}

/** One web-search result captured for the debug panel. */
export interface DebugResult {
  title: string;
  url: string;
  snippet: string;
}

/** Per-tool diagnostics for a turn (web search query/results/errors). */
export interface DebugEntry {
  tool: string;
  round?: number;
  query?: string;
  /** True when T1 decided a web search was warranted. */
  gated?: boolean;
  /** Which backend(s) answered: clock, brave, bing, duckduckgo, stub, none. */
  backend?: string | null;
  /** Per-backend outcome (e.g. "brave: not configured", "duckduckgo: 0 results"). */
  backendTrace?: string[];
  resultCount?: number;
  results?: DebugResult[];
  /** Non-fatal degradation (e.g. clock answered but live web search found nothing). */
  warning?: string | null;
  error?: string | null;
}

/** Round-local evidence returned by one executed skill. */
export interface RouterSkillEvidence {
  skill: string;
  kb?: string;
  resultCount?: number;
  backend?: string | null;
  error?: string | null;
}

/** One executed action selected by the post-ready T2 action phase. */
export interface ActionTile {
  action: string;
  status: string;
  detail?: string;
}

/** One planned/executed action's label and time-to-process, in milliseconds. */
export interface ActionTiming {
  /** Short label, e.g. `q3 → GaiaKB`, `q1 → Web`, `WhatsApp`, `Upsert GaiaKB`. */
  type: string;
  /** Milliseconds spent processing this one action (may be fractional). */
  ms: number;
}

/** Diagnostics for the pull pass (LLM Call 1 + retrieval), shown in the debug panel. */
export interface PullDebug {
  /** The model that produced LLM Call 1. */
  model: string;
  /** Wall-clock milliseconds spent in the LLM Call 1 request. */
  llmMs: number;
  /** The retrieval actions Call 1 chose this turn, each with the ms it took. */
  actions: ActionTiming[];
}

/** One planned push action's type and time-to-process, in milliseconds. */
export type PushActionTiming = ActionTiming;

/** Diagnostics for the push pass (LLM Call 2 + side effects), shown in the debug panel. */
export interface PushDebug {
  /** The model that produced LLM Call 2. */
  model: string;
  /** Wall-clock milliseconds spent in the LLM Call 2 request. */
  llmMs: number;
  /** One entry per planned side effect: its type and processing time. */
  actions: PushActionTiming[];
}

/** The real wall-clock latency of one Cosmos write this turn. */
export interface WriteTiming {
  /** Short label, e.g. `UsersDataLake`, `Upsert GaiaDiary`, `GaiaConnections delta`. */
  type: string;
  /** Wall-clock milliseconds the write took (may be fractional). */
  ms: number;
  /** True when the write succeeded, false when it failed. */
  ok: boolean;
}

/** Visible persistence status for the turn's mandatory Cosmos write-back. */
export interface WriteStatus {
  /** True when the turn was saved (or was an idempotent replay no-op). */
  ok: boolean;
  /** Confirmation on success (id, action, size), or the error detail on failure. */
  detail: string;
  /** Per-operation write latency, one entry per Cosmos write in execution order. */
  operations?: WriteTiming[];
}

/**
 * One entry in a turn's live process-log, streamed as the engine moves through
 * each phase so the UI can show where Gaia is right now (pull model, retrieval,
 * push model, persistence) and surface any warning or error immediately.
 */
export interface TurnEvent {
  /** Monotonic 0-based sequence number within the turn, for stable ordering. */
  seq: number;
  /** Phase that produced the event, e.g. `turn`, `pull`, `retrieval`, `push`, `persist`. */
  phase: string;
  /** Severity of the event. */
  level: 'info' | 'warn' | 'error';
  /** Human-readable description of what just happened. */
  message: string;
  /** Wall-clock milliseconds for the step this event closes, when timed. */
  ms?: number;
}

/** One ordered operation event emitted by Gaia internals. */
export interface OperationEvent {
  seq: number;
  ts: string;
  source: string;
  op: string;
  status: string;
  detail?: string;
  convId?: string;
  round?: number;
  /** Context-window size (in estimated tokens) fed to a think tier (T1/T2/T3). */
  tokens?: number;
}

/** One router loop round and the skills executed during it. */
export interface RouterRound {
  round: number;
  plannedSkills?: string[];
  skillsUsed: string[];
  skillEvidence?: RouterSkillEvidence[];
  ready: boolean;
}

/** Shape returned by POST /v1/conversations/{conv_id}/messages and the SSE `done` event. */
export interface ReplyResult {
  reply: string;
  verdict: Verdict;
  routing: Routing;
  /** Average of the six emotion scores (att). */
  attention: number;
  /** Emotional reading of the user's input across the six basic emotions. */
  emotions?: Emotions;
  thoughtId: string;
  /** Search actions taken during the turn (spec 04), e.g. WEB, MEMORY, KNOWLEDGE_BASE. */
  searches?: string[];
  /** Action phase outcomes (SEND_MESSAGE, SEND_PUSH, ADD_STM, ADD_LTM, ADD_KB). */
  actions?: ActionTile[];
  /** Short summary of the side effects Gaia planned this turn (WhatsApp / Push / Edwino / store writes). */
  actionsSummary?: string;
  /** Router rounds with per-round skill usage and readiness. */
  routerRounds?: RouterRound[];
  /** Per-tool diagnostics (web search query/results/errors). */
  debug?: DebugEntry[];
  /** Ordered operation flow for this turn. */
  flow?: OperationEvent[];
  /** Pull-pass diagnostics (model, llm time, actions chosen). */
  pullDebug?: PullDebug;
  /** Push-pass diagnostics (model, llm time, per-action type + time). */
  pushDebug?: PushDebug;
  /** Mandatory Cosmos write-back status; surfaced visibly so failures aren't silent. */
  write?: WriteStatus;
  /** Live process-log of every phase this turn entered, plus warnings/errors. */
  events?: TurnEvent[];
}

/** Subset of {@link ReplyResult} kept on a rendered message. */
export interface ReplyMeta {
  verdict: Verdict;
  routing: Routing;
  /** Average of the six emotion scores (att). */
  attention: number;
  /** Emotional reading of the user's input across the six basic emotions. */
  emotions?: Emotions;
  thoughtId: string;
  searches?: string[];
  actions?: ActionTile[];
  /** Short summary of the side effects Gaia planned this turn. */
  actionsSummary?: string;
  routerRounds?: RouterRound[];
  debug?: DebugEntry[];
  flow?: OperationEvent[];
  /** Pull-pass diagnostics (model, llm time, actions chosen). */
  pullDebug?: PullDebug;
  /** Push-pass diagnostics (model, llm time, per-action type + time). */
  pushDebug?: PushDebug;
  /** Mandatory Cosmos write-back status; surfaced visibly so failures aren't silent. */
  write?: WriteStatus;
  /** Live process-log of every phase this turn entered, plus warnings/errors. */
  events?: TurnEvent[];
}

/** Streaming frames delivered over the WebSocket transport (`streamWS`). */
export type StreamEvent =
  | { kind: 'token'; token: string }
  | { kind: 'done'; result: ReplyResult }
  | { kind: 'event'; event: TurnEvent }
  | { kind: 'error'; error: string };

/** Authenticated identity surfaced to the UI. */
export interface AuthUser {
  /** Subject claim — bound to the user's wing server-side. */
  sub: string;
  name?: string;
  email?: string;
  picture?: string;
  /** GitHub login (username), present only for GitHub sign-ins. */
  githubLogin?: string;
  /** Which provider this identity was issued by. */
  provider: 'google' | 'github';
  /** Bearer token sent on every API call (opaque Gaia session token). */
  token: string;
  /** Unix seconds when `token` expires. */
  expiresAt?: number;
  /** Long-lived refresh token used for silent renewal. */
  refreshToken?: string;
}
