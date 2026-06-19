<script lang="ts">
  import { mountGoogleButton } from '$lib/auth/google';
  import { auth, authMode } from '$lib/stores/auth';

  let name = $state('');
  let googleButtonEl: HTMLDivElement | null = $state(null);
  let googleError = $state<string | null>(null);
  const isDevMode = $derived($authMode === 'dev');

  function go() {
    auth.devSignIn(name);
  }

  function pickMode(mode: 'google' | 'dev') {
    auth.setMode(mode);
  }

  $effect(() => {
    if (!auth.canUseGoogle || isDevMode || !googleButtonEl) return;
    let cancelled = false;
    googleError = null;
    void mountGoogleButton(googleButtonEl, async (credential: string) => {
      if (cancelled) return;
      try {
        await auth.signInWithGoogleCredential(credential);
      } catch (e) {
        googleError = e instanceof Error ? e.message : String(e);
      }
    }).catch((e) => {
      if (!cancelled) googleError = e instanceof Error ? e.message : String(e);
    });

    return () => {
      cancelled = true;
      if (googleButtonEl) googleButtonEl.innerHTML = '';
    };
  });
</script>

<div class="wrap">
  <div class="card">
    <img class="logo" src="/icons/icon-192.png" alt="Gaia" width="88" height="88" />
    <h1>Gaia</h1>
    <p class="tag">Your private, always-on companion.</p>

    {#if auth.canUseGoogle}
      <div class="toggle" role="tablist" aria-label="Authentication mode">
        <button
          role="tab"
          aria-selected={!isDevMode}
          class:active={!isDevMode}
          onclick={() => pickMode('google')}
        >Google</button>
        <button
          role="tab"
          aria-selected={isDevMode}
          class:active={isDevMode}
          onclick={() => pickMode('dev')}
        >Dev</button>
      </div>
    {/if}

    {#if isDevMode}
      <p class="hint">
        Development mode. Pick a name to enter your wing.
      </p>
      <form
        onsubmit={(e) => {
          e.preventDefault();
          go();
        }}
      >
        <input
          bind:value={name}
          placeholder="Your name"
          aria-label="Your name"
          autocomplete="off"
        />
        <button type="submit" class="primary">Enter Gaia</button>
      </form>
    {:else}
      <div class="google-button" bind:this={googleButtonEl}></div>
      <p class="hint">Sign in with Google to enter your private wing.</p>
      {#if googleError}
        <p class="error">{googleError}</p>
      {/if}
    {/if}
  </div>
</div>

<style>
  .wrap {
    height: 100%;
    display: grid;
    place-items: center;
    padding: 24px;
  }
  .card {
    width: min(420px, 100%);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 28px;
    text-align: center;
    box-shadow: var(--shadow);
  }
  .logo {
    width: 88px;
    height: 88px;
    border-radius: 999px;
    border: 3px solid var(--accent);
    box-shadow: 0 0 28px rgba(124, 140, 255, 0.55);
  }
  h1 {
    margin: 6px 0 0;
    letter-spacing: 0.5px;
  }
  .tag {
    margin: 4px 0 20px;
    color: var(--text-dim);
  }
  .toggle {
    margin-bottom: 14px;
    display: flex;
    gap: 8px;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px;
  }
  .toggle button {
    flex: 1;
    border: none;
    border-radius: 999px;
    padding: 8px 10px;
    background: transparent;
    color: var(--text-dim);
    font: inherit;
    font-weight: 600;
    cursor: pointer;
  }
  .toggle button.active {
    color: #0b1020;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
  }
  form {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  input {
    padding: 12px 14px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--bg-elev-2);
    color: var(--text);
    font: inherit;
    outline: none;
  }
  input:focus {
    border-color: var(--accent);
  }
  .primary {
    padding: 12px 16px;
    border: none;
    border-radius: var(--radius);
    font: inherit;
    font-weight: 600;
    color: #0b1020;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
  }
  .google-button {
    display: flex;
    justify-content: center;
  }
  .error {
    margin-top: 8px;
    font-size: 13px;
    color: var(--danger);
    word-break: break-word;
  }
  .hint {
    margin: 14px 0 0;
    font-size: 13px;
    color: var(--text-dim);
  }
</style>
