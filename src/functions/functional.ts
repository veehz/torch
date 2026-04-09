import { Tensor } from '../tensor';
import { createOperation } from './registry';
import { ArgumentType } from './base';

function generate_function(opname: string) {
  return (...args: ArgumentType[]) => {
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

/**
 * @ignore
 * Get left index in a binary function
 */
export const __left_index__ = generate_binary_function('__left_index__');

/**
 * @ignore
 * Get right index in a binary function
 */
export const __right_index__ = generate_binary_function('__right_index__');

// binary pointwise

/**
 * Adds two tensors element-wise.
 */
export const add = generate_binary_function('add');

/**
 * Subtracts the second tensor from the first tensor element-wise.
 */
export const sub = generate_binary_function('sub');

/**
 * Multiplies two tensors element-wise.
 */
export const mul = generate_binary_function('mul');

/**
 * Divides the first tensor by the second tensor element-wise.
 */
export const div = generate_binary_function('div');

/**
 * Raises the first tensor to the power of the second tensor element-wise.
 */
export const pow = generate_binary_function('pow');

/**
 * Computes the element-wise remainder of the division of the first tensor by the second tensor.
 */
export const fmod = generate_binary_function('fmod');

/**
 * Returns the element-wise maximum of the two tensors.
 */
export const maximum = generate_binary_function('maximum');

/**
 * Returns the element-wise minimum of the two tensors.
 */
export const minimum = generate_binary_function('minimum');

// unary pointwise

/**
 * Computes the natural logarithm of the input tensor element-wise.
 */
export const log = generate_unary_function('log');

/**
 * Computes the square root of the input tensor element-wise.
 */
export const sqrt = generate_unary_function('sqrt');

/**
 * Computes the exponential of the input tensor element-wise.
 */
export const exp = generate_unary_function('exp');

/**
 * Computes the square of the input tensor element-wise.
 */
export const square = generate_unary_function('square');

/**
 * Computes the absolute value of the input tensor element-wise.
 */
export const abs = generate_unary_function('abs');

/**
 * Computes the sign of the input tensor element-wise.
 */
export const sign = generate_unary_function('sign');

/**
 * Negates the input tensor element-wise.
 */
export const neg = generate_unary_function('neg');

/**
 * Computes the reciprocal of the input tensor element-wise.
 */
export const reciprocal = generate_unary_function('reciprocal');

/**
 * Replaces NaN values in the input tensor with 0, positive infinity with a large finite number, and negative infinity with a small finite number.
 */
export const nan_to_num = generate_unary_function('nan_to_num');

/**
 * Reshapes the input tensor to the given shape.
 */
export const reshape = generate_function('reshape');

/**
 * Removes all dimensions of size 1 from the input tensor.
 */
export const squeeze = generate_function('squeeze');

/**
 * Adds a dimension of size 1 to the input tensor at the given position.
 */
export const unsqueeze = generate_function('unsqueeze');

/**
 * Expands the input tensor to the given shape.
 */
export const expand = generate_function('expand');

// trigonometric

/**
 * Computes the sine of the input tensor element-wise.
 */
export const sin = generate_unary_function('sin');

/**
 * Computes the cosine of the input tensor element-wise.
 */
export const cos = generate_unary_function('cos');

/**
 * Computes the tangent of the input tensor element-wise.
 */
export const tan = generate_unary_function('tan');

// reduction

/**
 * Computes the sum of the elements of the input tensor.
 */
export const sum = generate_function('sum');

/**
 * Computes the mean of the elements of the input tensor.
 */
export const mean = generate_function('mean');

/**
 * Computes the minimum of the elements of the input tensor.
 */
export const min = generate_function('min');

/**
 * Computes the maximum of the elements of the input tensor.
 */
export const max = generate_function('max');

// linalg

/**
 * Transposes the input tensor.
 */
export const transpose = generate_function('transpose');

/**
 * Computes the matrix product of the two input tensors.
 */
export const matmul = generate_binary_function('matmul');

// comparison

/**
 * Checks if the first tensor is less than the second tensor element-wise.
 */
export const lt = generate_binary_function('lt');

/**
 * Checks if the first tensor is greater than the second tensor element-wise.
 */
export const gt = generate_binary_function('gt');

/**
 * Checks if the first tensor is less than or equal to the second tensor element-wise.
 */
export const le = generate_binary_function('le');

/**
 * Checks if the first tensor is greater than or equal to the second tensor element-wise.
 */
export const ge = generate_binary_function('ge');

/**
 * Checks if the first tensor is equal to the second tensor element-wise.
 */
export const eq = generate_binary_function('eq');

/**
 * Checks if the first tensor is not equal to the second tensor element-wise.
 */
export const ne = generate_binary_function('ne');

/**
 * Checks if the two tensors are equal element-wise within a given tolerance.
 */
export function allclose(
  a: Tensor,
  b: Tensor,
  rtol: number = 1e-5,
  atol: number = 1e-8,
  equal_nan: boolean = false
): boolean {
  return a.allclose(b, rtol, atol, equal_nan);
}

/**
 * Returns the number of elements in the input tensor.
 */
export function numel(a: Tensor): number {
  return a.dataLength();
}

/**
 * Flattens the input tensor.
 */
export function flatten(input: Tensor, start_dim: number = 0, end_dim: number = -1): Tensor {
  return input.flatten(start_dim, end_dim);
}

/**
 * Concatenates tensors along a given dimension.
 */
export function cat(tensors: Tensor[], dim: number = 0): Tensor {
  const operation = createOperation('cat');
  return operation.forward(tensors, dim);
}

/**
 * Alias for {@link cat}.
 */
export const concatenate = cat;

/**
 * Alias for {@link cat}.
 */
export const concat = cat;

/**
 * Computes the softmax of the input tensor along the given dimension.
 */
export function softmax(input: Tensor, dim: number): Tensor {
  const operation = createOperation('softmax');
  return operation.forward(input, dim);
}

/**
 * Clamps all elements in input tensor to the range [min, max].
 */
export function clamp(input: Tensor, min: number, max: number): Tensor {
  const operation = createOperation('clamp');
  return operation.forward(input, min, max);
}

/**
 * Alias for {@link clamp}.
 */
export const clip = clamp;

/**
 * Stack tensors along a new dimension.
 */
export function stack(tensors: Tensor[], dim: number = 0): Tensor {
  return cat(tensors.map(t => t.unsqueeze(dim)), dim);
}
