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
  /** Router rounds with per-round skill usage and readiness. */
  routerRounds?: RouterRound[];
  /** Per-tool diagnostics (web search query/results/errors). */
  debug?: DebugEntry[];
  /** Ordered operation flow for this turn. */
  flow?: OperationEvent[];
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
  routerRounds?: RouterRound[];
  debug?: DebugEntry[];
  flow?: OperationEvent[];
}

/** SSE event frames emitted by /v1/conversations/{conv_id}/stream. */
export type StreamEvent =
  | { kind: 'token'; token: string }
  | { kind: 'done'; result: ReplyResult }
  | { kind: 'error'; error: string };

/** Authenticated identity surfaced to the UI. */
export interface AuthUser {
  /** Subject claim — bound to the user's wing server-side. */
  sub: string;
  name?: string;
  email?: string;
  picture?: string;
  /** Bearer token sent on every API call (Gaia session JWT in prod, sub in dev). */
  token: string;
  /** Unix seconds when `token` expires (Gaia session JWT). Omitted in dev mode. */
  expiresAt?: number;
  /** Long-lived refresh token used for silent renewal. Omitted in dev mode. */
  refreshToken?: string;
  /** True when running in local dev-subject mode. */
  dev: boolean;
}
