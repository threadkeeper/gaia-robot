/**
 * Gaia API client — talks to the conversation gateway (src/gaia/app/routes.py).
 *
 * Transports:
 *   - sendMessage:  POST /v1/conversations/{conv}/messages (non-streaming)
 *   - streamWS:     WS   /v1/ws/{conv} (bidirectional)
 */
import { apiUrl, wsUrl } from '$lib/config';
import type { OperationEvent, ReplyResult, StreamEvent } from '$lib/types';

export type DebugStreamEvent =
  | { kind: 'op'; op: OperationEvent }
  | { kind: 'ping' }
  | { kind: 'error'; error: string };

/** Session exchange returned by the auth endpoints (Google, GitHub, refresh). */
export interface AuthExchange {
  token: string;
  expiresAt: number;
  refreshToken: string;
  user: {
    sub: string;
    name?: string;
    email?: string;
    picture?: string;
    githubLogin?: string;
  };
}

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** Liveness check (no auth). */
export async function health(): Promise<boolean> {
  try {
    const r = await fetch(apiUrl('/healthz'), { method: 'GET' });
    return r.ok;
  } catch {
    return false;
  }
}

/** Non-streaming turn. Returns the full reply once safety + Merkle have passed. */
export async function sendMessage(
  convId: string,
  text: string,
  token: string,
  signal?: AbortSignal
): Promise<ReplyResult> {
  const res = await fetch(apiUrl(`/v1/conversations/${encodeURIComponent(convId)}/messages`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders(token) },
    body: JSON.stringify({ text }),
    signal
  });
  if (!res.ok) {
    throw new ApiError(res.status, await safeText(res));
  }
  return (await res.json()) as ReplyResult;
}

/** Exchange a Google ID token for a Gaia session token. */
export async function exchangeGoogleToken(idToken: string): Promise<AuthExchange> {
  const res = await fetch(apiUrl('/v1/auth/google'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ idToken })
  });
  if (!res.ok) {
    throw new ApiError(res.status, await safeText(res));
  }
  return (await res.json()) as AuthExchange;
}

/** Exchange a GitHub OAuth authorization code for a Gaia session token. */
export async function exchangeGithubCode(
  code: string,
  redirectUri?: string
): Promise<AuthExchange> {
  const res = await fetch(apiUrl('/v1/auth/github'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, redirectUri })
  });
  if (!res.ok) {
    throw new ApiError(res.status, await safeText(res));
  }
  return (await res.json()) as AuthExchange;
}

/** Exchange a Gaia refresh token for a fresh access token (silent renewal). */
export async function refreshSession(refreshToken: string): Promise<AuthExchange> {
  const res = await fetch(apiUrl('/v1/auth/refresh'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refreshToken })
  });
  if (!res.ok) {
    throw new ApiError(res.status, await safeText(res));
  }
  return (await res.json()) as AuthExchange;
}

/**
 * Stream Gaia operation events for the authenticated wing.
 */
export async function* streamDebugEvents(
  token: string,
  sinceSeq = 0,
  signal?: AbortSignal
): AsyncGenerator<DebugStreamEvent> {
  const url = apiUrl(`/v1/debug/stream?since_seq=${Math.max(0, Math.floor(sinceSeq))}`);
  const res = await fetch(url, {
    method: 'GET',
    headers: { Accept: 'text/event-stream', ...authHeaders(token) },
    signal
  });
  if (!res.ok || !res.body) {
    throw new ApiError(res.status, await safeText(res));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep: number;
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const evt = parseDebugFrame(frame);
        if (evt) yield evt;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseDebugFrame(frame: string): DebugStreamEvent | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  }
  const data = dataLines.join('\n');
  if (event === 'ping') return { kind: 'ping' };
  if (!data) return null;
  try {
    if (event === 'op') {
      return { kind: 'op', op: JSON.parse(data) as OperationEvent };
    }
  } catch {
    return { kind: 'error', error: `bad debug frame: ${data.slice(0, 120)}` };
  }
  return null;
}

/**
 * Stream a reply over a WebSocket. Mirrors the Web PubSub contract: the first
 * message is a hello carrying the auth token, then `{text}` is sent and the
 * server streams `{type:'token'}` frames followed by a `{type:'done'}` frame.
 */
export async function* streamWS(
  convId: string,
  text: string,
  token: string,
  signal?: AbortSignal
): AsyncGenerator<StreamEvent> {
  const socket = new WebSocket(wsUrl(`/v1/ws/${encodeURIComponent(convId)}`));
  const queue: StreamEvent[] = [];
  let resolveNext: (() => void) | null = null;
  let closed = false;
  let socketError: string | null = null;

  const wake = () => {
    if (resolveNext) {
      const r = resolveNext;
      resolveNext = null;
      r();
    }
  };

  signal?.addEventListener('abort', () => {
    try {
      socket.close();
    } catch {
      /* ignore */
    }
  });

  socket.onopen = () => {
    socket.send(JSON.stringify({ token }));
    socket.send(JSON.stringify({ text }));
  };
  socket.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'token') queue.push({ kind: 'token', token: String(msg.token ?? '') });
      else if (msg.type === 'done') {
        queue.push({ kind: 'done', result: msg.result as ReplyResult });
        socket.close();
      }
    } catch {
      queue.push({ kind: 'error', error: 'bad WS frame' });
    }
    wake();
  };
  socket.onerror = () => {
    socketError = 'websocket error';
    wake();
  };
  socket.onclose = () => {
    closed = true;
    wake();
  };

  try {
    for (;;) {
      while (queue.length) {
        const evt = queue.shift()!;
        yield evt;
        if (evt.kind === 'done') return;
      }
      if (socketError) {
        yield { kind: 'error', error: socketError };
        return;
      }
      if (closed) return;
      await new Promise<void>((resolve) => (resolveNext = resolve));
    }
  } finally {
    try {
      socket.close();
    } catch {
      /* ignore */
    }
  }
}

async function safeText(res: Response): Promise<string> {
  try {
    return (await res.text()) || res.statusText;
  } catch {
    return res.statusText;
  }
}
