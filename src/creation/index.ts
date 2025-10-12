import { Tensor } from '../tensor';

export function linspace(start: number, end: number, steps: number) {
  const data = [];
  const step = (end - start) / (steps - 1);
  for (let i = 0; i < steps - 1; i++) {
    data.push(start + i * step);
  }
  data.push(end);
  return new Tensor(data);
}

function get_shape_from_args(args: number[] | number[][]): number[] {
  if (Array.isArray(args[0])) {
    return args[0];
  }

  return args as number[];
}

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
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(Math.floor(Math.random() * (high - low) + low)));
  tensor.shape = shape;
  return tensor;
}