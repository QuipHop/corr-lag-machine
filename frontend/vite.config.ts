import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,          // 0.0.0.0 у контейнері
    port: 5173,
    strictPort: true,    // якщо порт зайнятий — не переключатися "тихо"
    hmr: {
      host: 'localhost', // браузер підключається саме сюди з хоста
      clientPort: 5173,  // публічний порт, проброшений з контейнера
      protocol: 'ws',
    },
  },
});
