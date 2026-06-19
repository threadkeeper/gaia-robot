<script lang="ts">
  import '../app.css';
  import { auth } from '$lib/stores/auth';
  import { onMount } from 'svelte';

  let { children } = $props();

  onMount(async () => {
    // Restore a previously persisted Gaia session (if any).
    try {
      await auth.restore();
    } catch {
      /* ignore — user can sign in manually */
    }

    // Register the service worker (generateSW build output) once mounted.
    if (!('serviceWorker' in navigator)) return;
    try {
      const { registerSW } = await import('virtual:pwa-register');
      registerSW({ immediate: true });
    } catch {
      // virtual module only exists in a PWA build; ignore in plain dev.
    }
  });
</script>

<div class="app">
  {@render children()}
</div>

<style>
  .app {
    height: 100dvh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
</style>
