<script lang="ts">
  import { browser } from '$app/environment';
  import Composer from '$lib/components/Composer.svelte';
  import InstallPrompt from '$lib/components/InstallPrompt.svelte';
  import MessageList from '$lib/components/MessageList.svelte';
  import SignIn from '$lib/components/SignIn.svelte';
  import TopBar from '$lib/components/TopBar.svelte';
  import { APP_VERSION } from '$lib/version';
  import { auth } from '$lib/stores/auth';
  import { conversation, conversationId } from '$lib/stores/conversation';

  const convId = conversationId();
  let hardResetInProgress = $state(false);

  function handleSend(text: string) {
    const user = $auth;
    if (!user) return;
    conversation.send(convId, text, (force) => auth.getValidToken(force));
  }

  async function clearClientArtifacts() {
    if (!browser) return;

    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }

    try {
      sessionStorage.clear();
    } catch {
      /* ignore */
    }

    const tasks: Promise<unknown>[] = [];

    if ('caches' in window) {
      tasks.push(
        caches.keys().then((keys) => Promise.allSettled(keys.map((key) => caches.delete(key))))
      );
    }

    if ('serviceWorker' in navigator) {
      tasks.push(
        navigator.serviceWorker
          .getRegistrations()
          .then((registrations) => Promise.allSettled(registrations.map((reg) => reg.unregister())))
      );
    }

    tasks.push(clearIndexedDbDatabases());

    clearCookies();
    await Promise.allSettled(tasks);
  }

  async function clearIndexedDbDatabases() {
    if (!browser || !('indexedDB' in window)) return;

    const indexedDb = window.indexedDB as IDBFactory & {
      databases?: () => Promise<Array<{ name?: string }>>;
    };

    if (typeof indexedDb.databases !== 'function') return;

    try {
      const databases = await indexedDb.databases();
      await Promise.allSettled(
        databases
          .map((db) => db.name)
          .filter((name): name is string => Boolean(name))
          .map((name) => deleteIndexedDbDatabase(name))
      );
    } catch {
      /* ignore */
    }
  }

  function deleteIndexedDbDatabase(name: string) {
    return new Promise<void>((resolve) => {
      const request = indexedDB.deleteDatabase(name);
      request.onsuccess = () => resolve();
      request.onerror = () => resolve();
      request.onblocked = () => resolve();
    });
  }

  function clearCookies() {
    if (!browser) return;
    const names = document.cookie
      .split(';')
      .map((part) => part.trim().split('=')[0])
      .filter(Boolean);

    if (!names.length) return;

    const domainParts = window.location.hostname.split('.').filter(Boolean);
    const domains = new Set<string>(['']);
    for (let i = 0; i < domainParts.length; i += 1) {
      const domain = domainParts.slice(i).join('.');
      domains.add(domain);
      domains.add(`.${domain}`);
    }

    const pathParts = window.location.pathname.split('/').filter(Boolean);
    const paths = new Set<string>(['/']);
    for (let i = 1; i <= pathParts.length; i += 1) {
      paths.add(`/${pathParts.slice(0, i).join('/')}`);
    }

    for (const name of names) {
      for (const path of paths) {
        document.cookie = `${name}=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Max-Age=0; Path=${path};`;
        for (const domain of domains) {
          if (!domain) continue;
          document.cookie = `${name}=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Max-Age=0; Path=${path}; Domain=${domain};`;
        }
      }
    }
  }

  async function hardReset() {
    if (!browser || hardResetInProgress) return;
    const confirmed = window.confirm(
      'This will clear cached Gaia data (cookies, storage, IndexedDB, service workers, and cache storage) and reload. Continue?'
    );
    if (!confirmed) return;

    hardResetInProgress = true;
    try {
      await clearClientArtifacts();
    } finally {
      const url = new URL(window.location.href);
      url.searchParams.set('fresh_install', Date.now().toString());
      window.location.replace(url.toString());
    }
  }
</script>

<svelte:head>
  <title>Gaia</title>
</svelte:head>

<div class="build-tools">
  <span class="version" aria-label={`App version ${APP_VERSION}`}>{APP_VERSION}</span>
  <button
    type="button"
    class="hard-reset"
    title="Clear all local Gaia/PWA data and reload"
    aria-label="Fresh install reset"
    disabled={hardResetInProgress}
    onclick={() => void hardReset()}
  >
    {hardResetInProgress ? '...' : '↻'}
  </button>
</div>

{#if !$auth}
  <SignIn />
{:else}
  <TopBar user={$auth} onclear={() => conversation.clear()} onsignout={() => auth.signOut()} />

  <div class="install">
    <InstallPrompt />
  </div>

  <MessageList messages={$conversation.messages} />

  <Composer
    busy={$conversation.busy}
    onsend={handleSend}
    oncancel={() => conversation.cancel()}
  />
{/if}

<style>
  .build-tools {
    position: fixed;
    top: max(8px, env(safe-area-inset-top));
    right: clamp(12px, 4vw, 40px);
    z-index: 40;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 6px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(11, 16, 32, 0.82);
    backdrop-filter: blur(8px);
  }
  .version {
    font-size: 11px;
    line-height: 1;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .hard-reset {
    width: 24px;
    height: 24px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-elev);
    color: var(--text);
    font-size: 13px;
    line-height: 1;
    display: grid;
    place-items: center;
    padding: 0;
  }
  .hard-reset:disabled {
    opacity: 0.6;
    cursor: wait;
  }
  .install {
    padding: 8px clamp(12px, 4vw, 40px) 0;
  }
  .install:empty {
    display: none;
  }

  @media (max-width: 640px) {
    .build-tools {
      top: max(6px, env(safe-area-inset-top));
      right: 10px;
      padding: 3px 5px;
      gap: 5px;
    }
    .version {
      font-size: 10px;
    }
    .hard-reset {
      width: 22px;
      height: 22px;
      font-size: 12px;
    }
  }
</style>
