/**
 * Conversation store — owns the message list for a single conversation and
 * drives a turn through the chosen streaming transport.
 */
import { browser } from '$app/environment';
import { ApiError, sendMessage, streamWS } from '$lib/api/client';
import { STREAM_TRANSPORT } from '$lib/config';
import type { ChatMessage, ReplyResult } from '$lib/types';
import { get, writable } from 'svelte/store';

/**
 * Resolves a valid bearer token, refreshing silently when needed. Returns
 * `null` when the session can no longer be renewed (user must sign in again).
 * Pass `force` to bypass the local validity check after a 401.
 */
export type TokenProvider = (force?: boolean) => Promise<string | null>;

function isUnauthorized(err: unknown): boolean {
  return err instanceof ApiError && err.status === 401;
}

function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

const CONV_KEY = 'gaia.convId.v1';

/** Stable per-device conversation id (sessionId = convId for ordered delivery). */
export function conversationId(): string {
  if (!browser) return 'conv-ssr';
  let id = localStorage.getItem(CONV_KEY);
  if (!id) {
    id = `conv-${crypto.randomUUID()}`;
    localStorage.setItem(CONV_KEY, id);
  }
  return id;
}

export interface ConversationState {
  messages: ChatMessage[];
  /** True while a turn is in flight. */
  busy: boolean;
}

function uid(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

function nowIso(): string {
  return new Date().toISOString();
}

function createConversation() {
  const store = writable<ConversationState>({ messages: [], busy: false });
  const { subscribe, update } = store;

  let controller: AbortController | null = null;

  function appendMessage(msg: ChatMessage) {
    update((s) => ({ ...s, messages: [...s.messages, msg] }));
  }

  function patchMessage(id: string, patch: Partial<ChatMessage>) {
    update((s) => ({
      ...s,
      messages: s.messages.map((m) => (m.id === id ? { ...m, ...patch } : m))
    }));
  }

  /** Send a user message and stream Gaia's reply into a placeholder bubble. */
  async function send(convId: string, text: string, getToken: TokenProvider): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed || get(store).busy) return;

    appendMessage({ id: uid('u'), role: 'user', text: trimmed, ts: nowIso() });

    const replyId = uid('g');
    appendMessage({ id: replyId, role: 'gaia', text: '', ts: nowIso(), streaming: true });
    update((s) => ({ ...s, busy: true }));

    controller = new AbortController();
    const signal = controller.signal;

    /**
     * Run a single turn with the given token. Streams into the reply bubble and
     * throws on failure (including {@link ApiError} 401) so the caller can decide
     * whether to refresh the token and retry.
     */
    async function runTurn(token: string): Promise<void> {
      let acc = '';
      const applyDone = (result: ReplyResult) =>
        patchMessage(replyId, {
          text: result.reply || acc,
          streaming: false,
          meta: {
            verdict: result.verdict,
            routing: result.routing,
            attention: result.attention,
            emotions: result.emotions,
            thoughtId: result.thoughtId,
            searches: result.searches,
            actions: result.actions,
            actionsSummary: result.actionsSummary,
            routerRounds: result.routerRounds,
            debug: result.debug,
            flow: result.flow,
            pullDebug: result.pullDebug,
            pushDebug: result.pushDebug,
            write: result.write
          }
        });

      try {
        if (STREAM_TRANSPORT === 'ws') {
          // Stream tokens over a WebSocket, applying each frame as it arrives.
          for await (const evt of streamWS(convId, trimmed, token, signal)) {
            if (evt.kind === 'token') {
              acc += evt.token;
              patchMessage(replyId, { text: acc });
            } else if (evt.kind === 'done') {
              applyDone(evt.result);
            } else if (evt.kind === 'error') {
              throw new Error(evt.error);
            }
          }
        } else {
          // Non-streaming transport: a single POST returns the full reply.
          const res = await sendMessage(convId, trimmed, token, signal);
          applyDone(res);
        }
      } catch (err) {
        if (isAbort(err)) return;
        // Let auth failures propagate so the caller can refresh + retry.
        if (isUnauthorized(err)) throw err;
        // Fall back to a non-streaming POST if streaming failed before any token.
        if (!acc) {
          const res = await sendMessage(convId, trimmed, token);
          applyDone(res);
          return;
        }
        throw err;
      }
    }

    try {
      const token = await getToken();
      if (!token) {
        patchMessage(replyId, { streaming: false, error: 'Please sign in to continue.' });
        return;
      }
      try {
        await runTurn(token);
      } catch (err) {
        if (isAbort(err)) return;
        if (isUnauthorized(err)) {
          // Token rejected — refresh once and retry the turn.
          const fresh = await getToken(true);
          if (!fresh) {
            patchMessage(replyId, {
              streaming: false,
              error: 'Your session expired. Please sign in again.'
            });
            return;
          }
          await runTurn(fresh);
          return;
        }
        throw err;
      }
    } catch (err) {
      if (!isAbort(err)) {
        patchMessage(replyId, { streaming: false, error: messageOf(err) });
      }
    } finally {
      controller = null;
      update((s) => ({ ...s, busy: false }));
    }
  }

  /** Cancel the in-flight turn, if any. */
  function cancel() {
    controller?.abort();
  }

  /** Clear the transcript (does not reset the conversation id). */
  function clear() {
    cancel();
    update(() => ({ messages: [], busy: false }));
  }

  return { subscribe, send, cancel, clear };
}

function messageOf(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

export const conversation = createConversation();
