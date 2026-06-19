<script lang="ts">
  let {
    busy = false,
    onsend,
    oncancel
  }: {
    busy?: boolean;
    onsend: (text: string) => void;
    oncancel: () => void;
  } = $props();

  let text = $state('');
  let textarea = $state<HTMLTextAreaElement | null>(null);

  function autosize() {
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  }

  function submit() {
    const t = text.trim();
    if (!t || busy) return;
    onsend(t);
    text = '';
    queueMicrotask(autosize);
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }
</script>

<form
  class="composer"
  onsubmit={(e) => {
    e.preventDefault();
    submit();
  }}
>
  <textarea
    bind:this={textarea}
    bind:value={text}
    oninput={autosize}
    onkeydown={onKeydown}
    rows="1"
    placeholder="Message Gaia…"
    aria-label="Message Gaia"
  ></textarea>

  {#if busy}
    <button type="button" class="btn stop" onclick={oncancel} aria-label="Stop generating">
      ■
    </button>
  {:else}
    <button type="submit" class="btn send" disabled={!text.trim()} aria-label="Send message">
      ↑
    </button>
  {/if}
</form>

<style>
  .composer {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    max-width: 900px;
    margin: 0 auto;
    width: 100%;
    padding: 10px clamp(12px, 4vw, 40px) max(10px, env(safe-area-inset-bottom));
  }
  textarea {
    flex: 1 1 auto;
    resize: none;
    min-height: 46px;
    max-height: 200px;
    padding: 12px 14px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--bg-elev);
    color: var(--text);
    font: inherit;
    line-height: 1.4;
    outline: none;
  }
  textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(124, 140, 255, 0.18);
  }
  .btn {
    flex: 0 0 auto;
    width: 46px;
    height: 46px;
    border-radius: 50%;
    border: none;
    font-size: 20px;
    color: #0b1020;
    display: grid;
    place-items: center;
    transition:
      transform 0.08s ease,
      opacity 0.15s ease;
  }
  .btn:active {
    transform: scale(0.92);
  }
  .send {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
  }
  .send:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .stop {
    background: var(--danger);
    color: #fff;
  }
</style>
