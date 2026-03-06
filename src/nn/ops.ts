import { UnaryFunctionMixin } from '../functions/mixin';

const Relu = UnaryFunctionMixin(
  (a: number[], x: number) => Math.max(a[x], 0),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(a.gt(0))));
  },
  "relu"
);

const Sigmoid = UnaryFunctionMixin(
  (a: number[], x: number) => 1 / (1 + Math.exp(-a[x])),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(a.exp().add(1).pow(-2).reciprocal().mul(a.exp()).mul(-1))));
  },
  "sigmoid"
);
