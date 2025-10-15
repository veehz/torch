import { Tensor } from '../tensor';
import { get_shape_from_args } from './utils';

/* TODO: use the correct distributions */

export function randn(...args: number[] | number[][]): Tensor {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(Math.random()));
  tensor.shape = shape;
  return tensor;
}

export function rand(...args: number[] | number[][]): Tensor {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(Math.random()));
  tensor.shape = shape;
  return tensor;
}

export function randint(low: number, high: number, shape: number[]): Tensor {
  const tensor = new Tensor(
    Array(shape.reduce((a, b) => a * b, 1)).fill(Math.floor(Math.random() * (high - low) + low))
  );
  tensor.shape = shape;
  return tensor;
}
