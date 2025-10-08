import { Tensor } from '../tensor';
import { getOperation } from './registry';

export function add(a: Tensor, b: Tensor): Tensor {
  const operation = new (getOperation('add'))();
  return operation.forward(a, b);
}

export function sub(a: Tensor, b: Tensor): Tensor {
  const operation = new (getOperation('sub'))();
  return operation.forward(a, b);
}

export function mul(a: Tensor, b: Tensor): Tensor {
  const operation = new (getOperation('mul'))();
  return operation.forward(a, b);
}

export function div(a: Tensor, b: Tensor): Tensor {
  const operation = new (getOperation('div'))();
  return operation.forward(a, b);
}

export function sum(a: Tensor): Tensor {
  const operation = new (getOperation('sum'))();
  return operation.forward(a);
}

export function pow(a: Tensor, b: Tensor): Tensor {
  const operation = new (getOperation('pow'))();
  return operation.forward(a, b);
}

export function log(a: Tensor): Tensor {
  const operation = new (getOperation('log'))();
  return operation.forward(a);
}
