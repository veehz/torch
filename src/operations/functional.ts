import { Tensor } from '../tensor';
import { getOperation } from './registry';

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

// binary pointwise

export const add = generate_binary_function('add');
export const sub = generate_binary_function('sub');
export const mul = generate_binary_function('mul');
export const div = generate_binary_function('div');
export const pow = generate_binary_function('pow');
export const fmod = generate_binary_function('fmod');

// unary pointwise

export const log = generate_unary_function('log');
export const sqrt = generate_unary_function('sqrt');
export const exp = generate_unary_function('exp');
export const abs = generate_unary_function('abs');
export const sign = generate_unary_function('sign');
export const neg = generate_unary_function('neg');
export const reciprocal = generate_unary_function('reciprocal');
export const reshape = generate_function('reshape');
export const unsqueeze = generate_function('unsqueeze');

// trigonometric

export const sin = generate_unary_function('sin');
export const cos = generate_unary_function('cos');
export const tan = generate_unary_function('tan');

// reduction

export const sum = generate_unary_function('sum');
export const mean = generate_unary_function('mean');

// linalg

export const transpose = generate_function('transpose');
export const matmul = generate_binary_function('matmul');

// comparison

export const lt = generate_binary_function('lt');
export const gt = generate_binary_function('gt');
export const le = generate_binary_function('le');
export const ge = generate_binary_function('ge');
export const eq = generate_binary_function('eq');
export const ne = generate_binary_function('ne');