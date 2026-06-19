import { sveltekit } from '@sveltejs/kit/vite';
import { SvelteKitPWA } from '@vite-pwa/sveltekit';
import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';

const BASE_VERSION = '0.2.0';

function getAppVersion(): string {
  const explicit = process.env.VITE_APP_VERSION ?? process.env.APP_VERSION;
  if (explicit) {
    return explicit.startsWith('v') ? explicit : `v${explicit}`;
  }

  try {
    const raw = execSync('git rev-list --count HEAD', {
      stdio: ['ignore', 'pipe', 'ignore']
    })
      .toString()
      .trim();
    const commitCount = Number.parseInt(raw, 10);
    return `v${BASE_VERSION}.${Number.isFinite(commitCount) && commitCount >= 0 ? commitCount : 0}`;
  } catch {
    return `v${BASE_VERSION}.0`;
  }
}

const appVersion = getAppVersion();

export default defineConfig({
  define: {
    __APP_VERSION__: JSON.stringify(appVersion)
  },
  plugins: [
    sveltekit(),
    SvelteKitPWA({
      // autoUpdate => generated SW uses skipWaiting + clientsClaim so already-installed
      // clients pick up a new build on next load instead of getting stuck on a "waiting"
      // worker (the old 'prompt' value left stale app shells serving outdated config).
      registerType: 'autoUpdate',
      injectRegister: null,
      strategies: 'generateSW',
      manifest: {
        name: 'Gaia',
        short_name: 'Gaia',
        description: 'Your private, always-on companion.',
        theme_color: '#0b1020',
        background_color: '#0b1020',
        display: 'standalone',
        orientation: 'portrait',
        start_url: '/',
        scope: '/',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
          { src: '/icons/maskable-192.png', sizes: '192x192', type: 'image/png', purpose: 'maskable' },
          { src: '/icons/maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
          { src: '/icons/apple-touch-icon-180.png', sizes: '180x180', type: 'image/png', purpose: 'any' }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,ico,woff,woff2}'],
        // Drop precaches from prior builds so a new worker doesn't keep serving a stale shell.
        cleanupOutdatedCaches: true,
        // Never cache the dynamic conversation API — always go to network.
        navigateFallbackDenylist: [/^\/v1\//, /^\/healthz/, /^\/readyz/],
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.startsWith('/v1/'),
            handler: 'NetworkOnly'
          }
        ]
      },
      devOptions: {
        enabled: false,
        type: 'module'
      }
    })
  ],
  server: {
    port: 5173,
    proxy: {
      // Dev: forward API + realtime to the local Rust backend on :8080 (or override via VITE_API_PROXY).
      '/v1': { target: process.env.VITE_API_PROXY ?? 'http://localhost:8080', changeOrigin: true, ws: true },
      '/healthz': { target: process.env.VITE_API_PROXY ?? 'http://localhost:8080', changeOrigin: true },
      '/readyz': { target: process.env.VITE_API_PROXY ?? 'http://localhost:8080', changeOrigin: true }
    }
  }
});
