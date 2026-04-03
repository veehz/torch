import { Tensor } from "../tensor";
import { createOperation } from "../functions/registry";

function generate_function(opname: string) {
  return (...args: (Tensor | number | number[] | null)[]) => {
    const operation = createOperation(opname);
    return operation.forward(...args);
  };
}

function generate_unary_function(opname: string) {
  return (a: Tensor | number) => {
    if (typeof a == 'number') {
      a = new Tensor(a);
    }

    const operation = createOperation(opname);
    return operation.forward(a);
  };
}

export const relu = generate_unary_function('relu');
export const sigmoid = generate_unary_function('sigmoid');

export const leaky_relu = generate_function('leaky_relu');

export const conv1d = generate_function('conv1d');
export const conv2d = generate_function('conv2d');
export const conv3d = generate_function('conv3d');
export const cross_entropy = generate_function('cross_entropy_loss');
export const nll_loss = generate_function('nll_loss');
export const max_pool2d = generate_function('max_pool2d');
