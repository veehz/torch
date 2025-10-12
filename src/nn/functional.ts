import { Tensor } from "../tensor";
import { getOperation } from "../operations/registry";

function generate_function(opname: string) {
  return (...args: (Tensor | number)[]) => {
    const operation = new (getOperation(opname))();
    return operation.forward(...args);
  };
}

function generate_unary_function(opname: string) {
  return (a: Tensor | number) => {
    if (typeof a == 'number') {
      a = new Tensor(a);
    }

    const operation = new (getOperation(opname))();
    return operation.forward(a);
  };
}

function generate_binary_function(opname: string) {
  return (a: Tensor | number, b: Tensor | number) => {
    if (typeof a == 'number') {
      a = new Tensor(a);
    }

    if (typeof b == 'number') {
      b = new Tensor(b);
    }

    const operation = new (getOperation(opname))();
    return operation.forward(a, b);
  };
}

export const relu = generate_unary_function('relu');
