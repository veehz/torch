import { Tensor } from '../tensor';
import { get_shape_from_args } from './utils';

export function ones(...args: number[] | number[][]): Tensor {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(1));
  tensor.shape = shape;
  return tensor;
}

export function zeros(...args: number[] | number[][]): Tensor {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(0));
  tensor.shape = shape;
  return tensor;
}

export function ones_like(tensor: Tensor): Tensor {
  return ones(tensor.shape);
}

export function zeros_like(tensor: Tensor): Tensor {
  return zeros(tensor.shape);
}
