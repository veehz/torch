import { defineConfig } from 'vite';
import path from 'path';
import dts from 'vite-plugin-dts';

export default defineConfig({
  plugins: [
    dts({
      outDir: 'build/types'
    })
  ],
  build: {
    // Avoid transforming/minifying function bodies used by gpu.js kernel parser
    minify: false,
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
    }
  },
  esbuild: {
    keepNames: true,
  },
  resolve: {
    alias: {
      'gpu.js': path.resolve(__dirname, './vendor/gpu-browser.min.js'),
    },
  },
});
