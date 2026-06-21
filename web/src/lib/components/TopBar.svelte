<script lang="ts">
  import { debug } from '$lib/stores/settings';
  import type { AuthUser } from '$lib/types';

  let {
    user,
    onclear,
    onsignout
  }: {
    user: AuthUser;
    onclear: () => void;
    onsignout: () => void;
  } = $props();

  let menuOpen = $state(false);
  const initial = $derived((user.name ?? user.sub).charAt(0).toUpperCase());
</script>

<header class="bar">
  <div class="brand">
    <span class="mark">◎</span>
    <span class="name">Gaia</span>
  </div>

  <div class="right">
    <button
      class="icon"
      class:active={$debug}
      title={$debug ? 'Hide debug info' : 'Show debug info'}
      aria-label="Toggle debug info"
      aria-pressed={$debug}
      onclick={() => debug.toggle()}
    >
      &#8505;
    </button>
    <button class="icon" title="New conversation" aria-label="Clear conversation" onclick={onclear}>
      ⟲
    </button>
    <div class="account">
      <button class="avatar" aria-label="Account menu" onclick={() => (menuOpen = !menuOpen)}>
        {#if user.picture}
          <img src={user.picture} alt="" />
        {:else}
          {initial}
        {/if}
      </button>
      {#if menuOpen}
        <div class="menu" role="menu">
          <div class="who">
            <strong>{user.name ?? user.sub}</strong>
            {#if user.email}<span>{user.email}</span>{/if}
            {#if user.githubLogin}<span class="badge">@{user.githubLogin}</span>{/if}
          </div>
          <button
            role="menuitem"
            onclick={() => {
              menuOpen = false;
              onsignout();
            }}>Sign out</button
          >
        </div>
      {/if}
    </div>
  </div>
</header>

<style>
  .bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px clamp(12px, 4vw, 40px);
    padding-top: max(10px, env(safe-area-inset-top));
    border-bottom: 1px solid var(--border);
    background: rgba(11, 16, 32, 0.7);
    backdrop-filter: blur(10px);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    letter-spacing: 0.3px;
  }
  .mark {
    color: var(--accent);
    font-size: 20px;
  }
  .right {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-right: 104px;
  }
  .icon {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 16px;
  }
  .icon.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #0b1020;
  }
  .account {
    position: relative;
  }
  .avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #0b1020;
    font-weight: 700;
    overflow: hidden;
    padding: 0;
  }
  .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .menu {
    position: absolute;
    right: 0;
    top: 46px;
    min-width: 200px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 8px;
    z-index: 20;
  }
  .who {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 6px;
    font-size: 13px;
  }
  .who span {
    color: var(--text-dim);
  }
  .badge {
    align-self: flex-start;
    margin-top: 4px;
    font-size: 11px;
    color: var(--accent-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 1px 8px;
  }
  .menu button {
    width: 100%;
    text-align: left;
    padding: 9px 10px;
    border: none;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text);
    font: inherit;
  }
  .menu button:hover {
    background: var(--bg-elev-2);
  }

  @media (max-width: 640px) {
    .right {
      margin-right: 92px;
      gap: 6px;
    }
  }
</style>
