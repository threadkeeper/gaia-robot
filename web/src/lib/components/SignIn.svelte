<script lang="ts">
  import { onMount } from 'svelte';
  import { consumeGithubRedirect, startGithubSignIn } from '$lib/auth/github';
  import { mountGoogleButton } from '$lib/auth/google';
  import { auth, authMode } from '$lib/stores/auth';

  let googleButtonEl: HTMLDivElement | null = $state(null);
  let googleError = $state<string | null>(null);
  let githubError = $state<string | null>(null);
  let githubBusy = $state(false);

  const isGithubMode = $derived($authMode === 'github');
  // Only offer a provider toggle when both sign-in methods are available.
  const showToggle = $derived(auth.canUseGoogle && auth.canUseGithub);

  function pickMode(mode: 'google' | 'github') {
    auth.setMode(mode);
  }

  function startGithub() {
    githubError = null;
    githubBusy = true;
    try {
      startGithubSignIn();
    } catch (e) {
      githubBusy = false;
      githubError = e instanceof Error ? e.message : String(e);
    }
  }

  // On load, finish any GitHub redirect leg (?code=&state=) and sign in.
  onMount(() => {
    if (!auth.canUseGithub) return;
    let redirect;
    try {
      redirect = consumeGithubRedirect();
    } catch (e) {
      githubError = e instanceof Error ? e.message : String(e);
      return;
    }
    if (!redirect) return;
    githubBusy = true;
    auth.setMode('github');
    void auth
      .signInWithGithubCode(redirect.code, redirect.redirectUri)
      .catch((e) => {
        githubError = e instanceof Error ? e.message : String(e);
      })
      .finally(() => {
        githubBusy = false;
      });
  });

  $effect(() => {
    if (!auth.canUseGoogle || isGithubMode || !googleButtonEl) return;
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

    {#if !auth.canUseGoogle && !auth.canUseGithub}
      <p class="hint">
        Sign-in is not configured for this build. Set <code>VITE_GOOGLE_CLIENT_ID</code>
        and/or <code>VITE_GITHUB_CLIENT_ID</code> to enable Google or GitHub sign-in.
      </p>
    {:else}
      {#if showToggle}
        <div class="toggle" role="tablist" aria-label="Authentication provider">
          <button
            role="tab"
            aria-selected={!isGithubMode}
            class:active={!isGithubMode}
            onclick={() => pickMode('google')}>Google</button
          >
          <button
            role="tab"
            aria-selected={isGithubMode}
            class:active={isGithubMode}
            onclick={() => pickMode('github')}>GitHub</button
          >
        </div>
      {/if}

      {#if isGithubMode || !auth.canUseGoogle}
        <button class="primary github" onclick={startGithub} disabled={githubBusy}>
          {githubBusy ? 'Connecting to GitHub…' : 'Continue with GitHub'}
        </button>
        <p class="hint">Sign in with GitHub to enter your private wing.</p>
        {#if githubError}
          <p class="error">{githubError}</p>
        {/if}
      {:else}
        <div class="google-button" bind:this={googleButtonEl}></div>
        <p class="hint">Sign in with Google to enter your private wing.</p>
        {#if googleError}
          <p class="error">{googleError}</p>
        {/if}
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
  .primary {
    width: 100%;
    padding: 12px 16px;
    border: none;
    border-radius: var(--radius);
    font: inherit;
    font-weight: 600;
    color: #0b1020;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    cursor: pointer;
  }
  .primary:disabled {
    opacity: 0.6;
    cursor: progress;
  }
  .primary.github {
    color: #fff;
    background: #24292f;
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
