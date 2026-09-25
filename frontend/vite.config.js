import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// ponytail: Tailwind v4's Vite plugin instead of postcss.config.js + autoprefixer.
// Two fewer files and two fewer dependencies for the same output.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Built assets are served by scope3_server.py from the same origin, so the
  // base has to be relative rather than absolute.
  base: './',
  build: { outDir: 'dist' },
  server: {
    port: 5173,
    proxy: { '/api/scope3': 'http://127.0.0.1:5004' },
  },
});
