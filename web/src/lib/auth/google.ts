/**
 * Google Identity Services helper for direct browser sign-in.
 *
 * This module renders the Google sign-in button and returns a Google ID token
 * credential, which the app exchanges for a Gaia session token via
 * POST /v1/auth/google.
 */
import { browser } from '$app/environment';
import { GOOGLE_CLIENT_ID } from '$lib/config';

type GoogleCredentialResponse = {
  credential?: string;
};

type GoogleAccountsId = {
  initialize: (opts: {
    client_id: string;
    callback: (response: GoogleCredentialResponse) => void;
    auto_select?: boolean;
    cancel_on_tap_outside?: boolean;
  }) => void;
  renderButton: (
    parent: HTMLElement,
    opts: {
      type?: 'standard' | 'icon';
      theme?: 'outline' | 'filled_blue' | 'filled_black';
      size?: 'large' | 'medium' | 'small';
      text?: 'signin_with' | 'signup_with' | 'continue_with' | 'signin';
      shape?: 'rectangular' | 'pill' | 'circle' | 'square';
      width?: number;
      logo_alignment?: 'left' | 'center';
    }
  ) => void;
};

type GoogleGlobal = {
  accounts: { id: GoogleAccountsId };
};

declare global {
  interface Window {
    google?: GoogleGlobal;
  }
}

let scriptPromise: Promise<void> | null = null;

async function loadScript(): Promise<void> {
  if (!browser) throw new Error('Google sign-in is only available in the browser.');
  if (window.google?.accounts?.id) return;
  if (scriptPromise) return scriptPromise;

  scriptPromise = new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>('script[data-gaia-google="1"]');
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error('Failed to load Google script.')), {
        once: true
      });
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    script.dataset.gaiaGoogle = '1';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load Google script.'));
    document.head.appendChild(script);
  });

  return scriptPromise;
}

export async function mountGoogleButton(
  parent: HTMLElement,
  onCredential: (idToken: string) => void | Promise<void>
): Promise<void> {
  if (!GOOGLE_CLIENT_ID) {
    throw new Error('VITE_GOOGLE_CLIENT_ID is not configured.');
  }
  await loadScript();

  const gsi = window.google?.accounts?.id;
  if (!gsi) {
    throw new Error('Google Identity Services is unavailable.');
  }

  gsi.initialize({
    client_id: GOOGLE_CLIENT_ID,
    callback: (response: GoogleCredentialResponse) => {
      const credential = response?.credential ?? '';
      if (!credential) return;
      void onCredential(credential);
    },
    auto_select: false,
    cancel_on_tap_outside: true
  });

  parent.innerHTML = '';
  gsi.renderButton(parent, {
    type: 'standard',
    theme: 'outline',
    size: 'large',
    text: 'continue_with',
    shape: 'pill',
    logo_alignment: 'left',
    width: 320
  });
}
