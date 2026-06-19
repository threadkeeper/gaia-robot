/**
 * UI settings — small, device-local preferences persisted to localStorage.
 */
import { browser } from '$app/environment';
import { writable } from 'svelte/store';

const DEBUG_KEY = 'gaia.debug.v1';

function createBoolStore(storageKey: string) {
  const initial = browser && localStorage.getItem(storageKey) === '1';
  const { subscribe, set, update } = writable<boolean>(initial);

  function persist(value: boolean) {
    if (browser) localStorage.setItem(storageKey, value ? '1' : '0');
  }

  return {
    subscribe,
    set(value: boolean) {
      persist(value);
      set(value);
    },
    toggle() {
      update((v) => {
        const next = !v;
        persist(next);
        return next;
      });
    }
  };
}

/** When true, message bubbles show the routing/safety/search debug chips. */
export const debug = createBoolStore(DEBUG_KEY);
