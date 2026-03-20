import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  build: {
    minify: false, // don't minify for npm package
    sourcemap: true,
    lib: {
      entry: path.resolve(__dirname, 'src/index.ts'),
      fileName: (format) => format === 'es' ? 'torch.node.es.mjs' : 'torch.node.cjs',
      formats: ['es', 'cjs']
    },
    outDir: 'build/node',
    target: 'node20',
  },
});
