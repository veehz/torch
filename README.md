# torch
machine-learning libraries for Source Academy

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

### Node

See `examples/basic_backpropagation.js`.

```bash
node examples/basic_backpropagation.js
```

### Browser

You can run `http-server` and load `examples/browser/index.html` to see how it works.

```bash
npx http-server --cors=* -c-1 .
# and navigate to http://localhost:8080/examples/browser/index.html to run torch in js
# or http://localhost:8080/examples/pyodide/index.html to run in python
```