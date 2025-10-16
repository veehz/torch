import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: path.resolve(__dirname, 'src/index.ts'),
      fileName: (format) => format === 'es' ? 'torch.node.es.mjs' : 'torch.node.cjs',
      formats: ['es', 'cjs']
    },
    rollupOptions: {
      // Don't bundle 'gpu.js'. The user's Node project will provide it.
      external: ['@veehz/gpu.js']
    },
    outDir: 'build/node',
    target: 'node20',
    minify: false,
  },
});
