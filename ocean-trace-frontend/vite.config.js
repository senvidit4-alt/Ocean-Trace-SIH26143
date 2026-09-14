import { defineConfig } from "vite";

// Plain static frontend — no framework, no bundler-specific features used.
// Vite here is only for a fast local dev server + a production build/minify step.
export default defineConfig({
  root: ".",
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "")
      }
    }
  },

  build: {
    outDir: "dist",
    emptyOutDir: true
  }
});
