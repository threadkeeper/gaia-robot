<script lang="ts">
  import type { ChatMessage } from '$lib/types';
  import ChatMessageView from './ChatMessage.svelte';

  let { messages }: { messages: ChatMessage[] } = $props();

  let viewport = $state<HTMLDivElement | null>(null);

  // Autoscroll to the newest message whenever the list changes.
  $effect(() => {
    // touch length + last text so streaming updates also scroll
    void messages.length;
    void messages.at(-1)?.text;
    if (viewport) {
      viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
    }
  });
</script>

<div class="viewport" bind:this={viewport}>
  {#if messages.length === 0}
    <div class="empty">
      <img class="logo" src="/icons/icon-192.png" alt="Gaia" width="96" height="96" />
      <h2>Hi, I'm Gaia.</h2>
      <p>Ask me anything. Your conversation stays in your private wing.</p>
    </div>
  {:else}
    <div class="list">
      {#each messages as message (message.id)}
        <ChatMessageView {message} />
      {/each}
    </div>
  {/if}
</div>

<style>
  .viewport {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 16px clamp(12px, 4vw, 40px);
  }
  .list {
    max-width: 900px;
    margin: 0 auto;
    padding-bottom: 8px;
  }
  .empty {
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: var(--text-dim);
    gap: 6px;
  }
  .empty .logo {
    width: 96px;
    height: 96px;
    border-radius: 999px;
    border: 3px solid var(--accent);
    box-shadow: 0 0 24px rgba(124, 140, 255, 0.5);
  }
  .empty h2 {
    margin: 8px 0 0;
    color: var(--text);
  }
  .empty p {
    margin: 0;
    max-width: 34ch;
  }
</style>
