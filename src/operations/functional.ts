import { Tensor } from '../tensor';
import { getOperation } from './registry';

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

export const add = generate_binary_function('add');
export const sub = generate_binary_function('sub');
export const mul = generate_binary_function('mul');
export const div = generate_binary_function('div');
export const sum = generate_unary_function('sum');
export const mean = generate_unary_function('mean');
export const pow = generate_unary_function('pow');
export const log = generate_unary_function('log');
export const matmul = generate_binary_function('matmul');
