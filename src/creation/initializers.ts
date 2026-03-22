import { Tensor, NestedNumberArray } from '../tensor';
import { _get_shape_from_args, _numel } from '../util';

export function tensor(data: NestedNumberArray, requires_grad: boolean = false): Tensor {
  return new Tensor(data, { requires_grad });
}

export function full(shape: number[], fill_value: number): Tensor {
  const t = new Tensor(Array(_numel(shape)).fill(fill_value));
  t.shape = shape;
  return t;
}

export function zeros(...args: number[] | number[][]): Tensor {
  return full(_get_shape_from_args(args), 0);
}

export function ones(...args: number[] | number[][]): Tensor {
  return full(_get_shape_from_args(args), 1);
}

export function empty(...args: number[] | number[][]): Tensor {
  return full(_get_shape_from_args(args), 0);
}

export function full_like(input: Tensor, fill_value: number): Tensor {
  return full(input.shape, fill_value);
}

export function zeros_like(input: Tensor): Tensor {
  return full(input.shape, 0);
}

export function ones_like(input: Tensor): Tensor {
  return full(input.shape, 1);
}

export function empty_like(input: Tensor): Tensor {
  return full(input.shape, 0);
}
