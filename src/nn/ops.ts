import { UnaryFunctionMixin } from '../functions/mixin';

const Relu = UnaryFunctionMixin(
  (a: number[], x: number) => Math.max(a[x], 0),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.gt(0)));
  },
  "relu"
);

const Sigmoid = UnaryFunctionMixin(
  (a: number[], x: number) => 1 / (1 + Math.exp(-a[x])),
  (a, aFn, dz) => {
    const res = a.sigmoid();
    aFn.backward(res.mul(res.mul(-1).add(1)).mul(dz));
  },
  "sigmoid"
);
