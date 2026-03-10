import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  build: {
    minify: true,
    sourcemap: true,
    lib: {
      entry: path.resolve(__dirname, 'src/index.ts'),
      name: 'torch',
      fileName: (format) => `torch.browser.${format}.js`,
      formats: ['es', 'umd']
    },
    outDir: 'build/browser',
    rollupOptions: {
      treeshake: false,
      output: {
        compact: false,
        inlineDynamicImports: false,
        minifyInternalExports: false,
      }
    },
  },
  esbuild: {
    keepNames: true,
  }
});
