<script lang="ts">
  import { onMount } from 'svelte';

  interface BeforeInstallPromptEvent extends Event {
    prompt: () => Promise<void>;
    userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
  }

  let deferred = $state<BeforeInstallPromptEvent | null>(null);
  let visible = $state(false);

  onMount(() => {
    const handler = (e: Event) => {
      e.preventDefault();
      deferred = e as BeforeInstallPromptEvent;
      visible = true;
    };
    window.addEventListener('beforeinstallprompt', handler);
    window.addEventListener('appinstalled', () => (visible = false));
    return () => window.removeEventListener('beforeinstallprompt', handler);
  });

  async function install() {
    if (!deferred) return;
    await deferred.prompt();
    await deferred.userChoice;
    deferred = null;
    visible = false;
  }
</script>

{#if visible}
  <div class="bar" role="dialog" aria-label="Install Gaia">
    <span>Install Gaia for a full-screen, app-like experience.</span>
    <div class="actions">
      <button class="ghost" onclick={() => (visible = false)}>Not now</button>
      <button class="primary" onclick={install}>Install</button>
    </div>
  </div>
{/if}

<style>
  .bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin: 0 auto;
    max-width: 900px;
    padding: 10px 16px;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 14px;
  }
  .actions {
    display: flex;
    gap: 8px;
  }
  button {
    border-radius: var(--radius-sm);
    padding: 7px 12px;
    font: inherit;
    border: 1px solid var(--border);
  }
  .ghost {
    background: transparent;
    color: var(--text-dim);
  }
  .primary {
    border: none;
    color: #0b1020;
    font-weight: 600;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
  }
</style>
