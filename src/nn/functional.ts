import { Tensor } from "../tensor";
import { getOperation } from "../functions/registry";

function generate_function(opname: string) {
  return (...args: (Tensor | number | number[] | null)[]) => {
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
export const sigmoid = generate_unary_function('sigmoid');

export const conv1d = generate_function('conv1d');
export const conv2d = generate_function('conv2d');
export const conv3d = generate_function('conv3d');
