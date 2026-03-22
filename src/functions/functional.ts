import { Tensor } from '../tensor';
import { createOperation } from './registry';

function generate_function(opname: string) {
  return (...args: (Tensor | number)[]) => {
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

function generate_binary_function(opname: string) {
  return (a: Tensor | number, b: Tensor | number) => {
    if (typeof a == 'number') {
      a = new Tensor(a);
    }

    if (typeof b == 'number') {
      b = new Tensor(b);
    }

    const operation = createOperation(opname);
    return operation.forward(a, b);
  };
}

// debug operations

export const __left_index__ = generate_binary_function('__left_index__');
export const __right_index__ = generate_binary_function('__right_index__');

// binary pointwise

export const add = generate_binary_function('add');
export const sub = generate_binary_function('sub');
export const mul = generate_binary_function('mul');
export const div = generate_binary_function('div');
export const pow = generate_binary_function('pow');
export const fmod = generate_binary_function('fmod');
export const maximum = generate_binary_function('maximum');
export const minimum = generate_binary_function('minimum');

// unary pointwise

export const log = generate_unary_function('log');
export const sqrt = generate_unary_function('sqrt');
export const exp = generate_unary_function('exp');
export const square = generate_unary_function('square');
export const abs = generate_unary_function('abs');
export const sign = generate_unary_function('sign');
export const neg = generate_unary_function('neg');
export const reciprocal = generate_unary_function('reciprocal');
export const nan_to_num = generate_unary_function('nan_to_num');
export const reshape = generate_function('reshape');
export const squeeze = generate_function('squeeze');
export const unsqueeze = generate_function('unsqueeze');
export const expand = generate_function('expand');

// trigonometric

export const sin = generate_unary_function('sin');
export const cos = generate_unary_function('cos');
export const tan = generate_unary_function('tan');

// reduction

export const sum = generate_function('sum');
export const mean = generate_function('mean');
export const min = generate_function('min');
export const max = generate_function('max');

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

export function allclose(
  a: Tensor,
  b: Tensor,
  rtol: number = 1e-5,
  atol: number = 1e-8,
  equal_nan: boolean = false
): boolean {
  return a.allclose(b, rtol, atol, equal_nan);
}

export function numel(a: Tensor): number {
  return a.dataLength();
}

export function flatten(input: Tensor, start_dim: number = 0, end_dim: number = -1): Tensor {
  return input.flatten(start_dim, end_dim);
}
