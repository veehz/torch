import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  build: {
    minify: 'esbuild',
    sourcemap: true,
    lib: {
      entry: path.resolve(__dirname, 'src/index.ts'),
      name: 'torch',
      fileName: () => `torch.min.js`,
      formats: ['umd']
    },
    emptyOutDir: false,
    outDir: 'build',
  },
  esbuild: {
    keepNames: true,
  }
});
