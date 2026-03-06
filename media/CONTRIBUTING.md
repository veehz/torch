## Codebase Structure

- [`src`](src)
    - [`index.ts`](src/index.ts) is the entry point of the library.
    - [`tensor.ts`](src/tensor.ts) is the main tensor class.
    - [`function`](function) contains all functions that tensors can perform.
    - [`nn`](nn) contains all neural network modules (for everything under `torch.nn`).
    - [`optim`](optim) contains all optimizers (for everything under `torch.optim`).
    - [`creation`](creation) contains all tensor creation functions (all functions that create a tensor not from scratch, including `zeros`, `randn`).
- [`examples`](examples) contains example usages of the library, including on node, on the browser, and using pyodide on the browser.
- [`test`](test) contains the test cases of the library, including on node and on the browser. See [Testing](#testing).

### Development Scripts

You can use `yarn watch` to automatically test after each edit.

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
