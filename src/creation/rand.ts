import { Tensor } from '../tensor';
import { _get_shape_from_args, _numel } from '../util';
import { uniformDist, normalDist } from '../prng';

export function randn(...args: number[] | number[][]): Tensor {
  const shape = _get_shape_from_args(args);
  const tensor = new Tensor(Array.from({ length: _numel(shape) }, normalDist()));
  tensor.shape = shape;
  return tensor;
}

export function rand(...args: number[] | number[][]): Tensor {
  const shape = _get_shape_from_args(args);
  const tensor = new Tensor(Array.from({ length: _numel(shape) }, uniformDist()));
  tensor.shape = shape;
  return tensor;
}

export function randint(low: number, high: number, shape: number[]): Tensor {
  const tensor = new Tensor(
    Array.from({ length: _numel(shape) }, () => Math.floor(uniformDist(low, high)()))
  );
  tensor.shape = shape;
  return tensor;
}

export function randperm(n: number): Tensor {
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < n; i++) {
    const j = Math.floor(uniformDist()() * (n - i)) + i;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  const tensor = new Tensor(arr);
  return tensor;
}
