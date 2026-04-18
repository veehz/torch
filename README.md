# torch
machine-learning libraries for Source Academy

The primary objective of this project is to create a reimplementation of PyTorch in TypeScript, with an educational focus. This project is developed with Source Academy integration in mind.

This project reimplements core parts of PyTorch while trying to keep the codebase simple, and the API as close to PyTorch as possible.

Using Pyodide, we can run Python code in the browser. Using `pyodide_bridge.py` in a way similar to `examples/pyodide/` we can run PyTorch-like code in the browser.

## Notable differences with PyTorch

- This library exposes extra information for debuggers and visualizers to catch, as seen in `events` in [`src/util.ts`](src/util.ts). It is similar to hooks in PyTorch.
- This library does not differentiate between LongTensors and FloatTensors. It uses `number` for all tensor elements.
- This library does not currently support devices, such as GPUs.

## Getting Started

Install yarn:

``` bash
corepack enable
# or
npm install -g yarn
```

Install dependencies:

```bash
yarn install
```

## Demo Usage

First, build it:

```bash
yarn build
```

See [examples/](examples/) for examples.

### Node

See `examples/basic_backpropagation.js`.

```bash
node examples/basic_backpropagation.js
```

### Browser

You can run `http-server` and load `examples/browser/index.html` to see how it works.

```bash
yarn serve
# and navigate to http://localhost:8080/examples/browser/index.html to run torch in js
# or http://localhost:8080/examples/pyodide/index.html to run in python
# or http://localhost:8080/test/ to run the tests
```

## Contributing

Contributions are welcome. The short version:

1. Run `yarn test` to verify everything passes.
2. Add tests for new ops or behaviour changes.
3. Follow the existing patterns — new ops go in [src/functions/ops.ts](src/functions/ops.ts).

For full details on the codebase, how to add operations, and the testing setup, see [CONTRIBUTING.md](CONTRIBUTING.md).
