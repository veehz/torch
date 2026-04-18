## Development Commands

```bash
yarn install          # Install dependencies
yarn build            # Build for browser and node (runs clean first)
yarn test             # Run tests against src/ (uses mocha + tsx)
yarn test:build       # Run tests against built library in build/
yarn test:coverage    # Run tests with coverage report
yarn test:watch       # Watch mode for tests (alias: yarn watch)
yarn lint             # Lint src/ with ESLint
yarn serve            # Serve on localhost:8080
yarn docs             # Build TypeDoc documentation
yarn update-tests     # Regenerate test/testcases.gen.js from scripts/generate_tests.py
```

To run a single test file:
```bash
yarn mocha --node-option conditions=torch-src test/tensor.test.js
```

## Codebase Structure

- [`src`](src)
    - [`index.ts`](src/index.ts) is the entry point of the library.
    - [`tensor.ts`](src/tensor.ts) is the main tensor class.
    - [`functions`](functions) contains all functions that tensors can perform.
    - [`nn`](nn) contains all neural network modules (for everything under `torch.nn`).
    - [`optim`](optim) contains all optimizers (for everything under `torch.optim`).
    - [`creation`](creation) contains all tensor creation functions (all functions that create a tensor not from scratch, including `zeros`, `randn`).
- [`examples`](examples) contains example usages of the library, including on node, on the browser, and using pyodide on the browser.
- [`test`](test) contains the test cases of the library, including on node and on the browser. See [Testing](#testing).

### Development Scripts

Use `yarn watch` (or `yarn test:watch`) to automatically re-run tests on each edit.

### Adding a new Function

To add a new function, add it to [`src/functions/ops.ts`](src/functions/ops.ts).

- To allow for `torch.<opname>(<tensor>, <args>)`, add it as well to [`src/functions/functional.ts`](src/functions/functional.ts).
- To allow for `<tensor>.<opname>(<args>)`, add it as well to [`src/tensor.ts`](src/tensor.ts) as a `tensor` method.

## Testing

Tests are run using `mocha`.

- Node: To test on node, run `yarn test`.
- Browser: To test on browser, run `yarn serve` (after `yarn build` if necessary) and navigate to http://localhost:8080/test/.

To create a new test:

1. Create a new `.test.js`/`.test.ts` file in [`test`](test) or write your new test in one of the existing files.
2. If you created a new file and would like to test it on the browser, add it to [`test/index.html`](test/index.html).

## Documentation

To see docs, run `yarn docs` to build, and run `yarn serve docs` to serve docs on http://localhost:8080/.

To ensure familiarity with PyTorch, docs should be derived from the PyTorch docs, as mentioned in [`NOTICE`](NOTICE).
