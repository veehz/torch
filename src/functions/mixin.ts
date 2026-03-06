import { Tensor } from '../tensor';
import {
  _broadcast_shape,
  _get_original_index_from_transposed_index,
  _get_original_index,
  _get_original_index_kernel,
  _pad_shape
} from '../broadcasting';
import { TorchFunction, BinaryFunction, UnaryFunction, nullOp } from './base';
import { registerOperation } from './registry';

export function BinaryFunctionMixin(
  operation: (a: number[], b: number[], a_index: number, b_index: number) => number,
  backward_operations: (a, b, aFn, bFn, dz) => void,
  opName: string | null = null
): typeof BinaryFunction {
  const kernel = (a: number[], as: number[], b: number[], bs: number[], bcs: number[], output_size: number) => {
    const res = Array(output_size);
    for (let x = 0; x < output_size; x++) {
      const a_index = _get_original_index(as, bcs, x);
      const b_index = _get_original_index(bs, bcs, x);
      res[x] = operation(a, b, a_index, b_index);
    }
    return res;
  }

  const forward_tensor = (a: Tensor, b: Tensor, operation: TorchFunction | null = null): Tensor => {
    const broadcast_shape = _broadcast_shape(a.shape, b.shape);
    const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
    const padded_b_shape = _pad_shape(b.shape, broadcast_shape);

    const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);

    return new Tensor(
      kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size) as number[],
      { requires_grad: a.requires_grad || b.requires_grad },
      { operation: operation, shape: broadcast_shape }
    );
  }

  const result = class extends BinaryFunction {
    protected _forward(a: Tensor, b: Tensor): Tensor {
      if (a.requires_grad || b.requires_grad) {
        this.saved_tensors = [a, b];
      }
      this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
      this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
      return forward_tensor(a, b, a.requires_grad || b.requires_grad ? this : null)
    }

    protected _backward(dz: Tensor): void {
      const [a, b] = this.saved_tensors;
      const [aFn, bFn] = this.next_functions;

      backward_operations(a, b, aFn, bFn, dz);
    }
  }
  if (opName) {
    registerOperation(opName, result);
  }
  return result;
}

export function UnaryFunctionMixin(
  operation: (a: number[], x: number) => number,
  backward_operations: (a, aFn, dz) => void,
  opName: string | null = null
): typeof UnaryFunction {
  const kernel = (a: number[], output_size: number) => {
    const res = Array(output_size);
    for (let x = 0; x < output_size; x++) {
      res[x] = operation(a, x);
    }
    return res;
  };
  const forward_tensor = (a: Tensor, operation: TorchFunction | null = null): Tensor => {
      const output_size = a.dataLength();

      return new Tensor(
        kernel(a.data, output_size) as number[],
        { requires_grad: a.requires_grad },
        { operation: operation, shape: a.shape }
      );
    }

  const result = class extends UnaryFunction {
    protected _forward(a: Tensor): Tensor {
      if (a.requires_grad) {
        this.saved_tensors = [a];
      }
      this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
      return forward_tensor(a, a.requires_grad ? this : null)
    }

    protected _backward(dz: Tensor): void {
      const [a] = this.saved_tensors;
      const [aFn] = this.next_functions;

      backward_operations(a, aFn, dz);
    }
  }
  if (opName) {
    registerOperation(opName, result);
  }
  return result;
}
