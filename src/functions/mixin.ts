import { Tensor } from '../tensor';
import {
  _broadcast_shape,
  _get_original_index,
  _pad_shape
} from '../broadcasting';
import { TorchFunction, BinaryFunction, UnaryFunction, nullOp, resultRequiresGrad } from './base';
import { registerOperation } from './registry';
import { _get_reduction_shape, _get_strides, _ravel_index, _unravel_index } from './util';

export function BinaryFunctionMixin(
  operation: (a: number[], b: number[], a_index: number, b_index: number) => number,
  backward_operations: (a, b, aFn, bFn, dz) => void,
  opName: string | null = null
): typeof BinaryFunction {
  const kernel = (
    a: number[],
    as: number[],
    b: number[],
    bs: number[],
    bcs: number[],
    output_size: number
  ) => {
    const res = Array(output_size);
    for (let x = 0; x < output_size; x++) {
      const a_index = _get_original_index(as, bcs, x);
      const b_index = _get_original_index(bs, bcs, x);
      res[x] = operation(a, b, a_index, b_index);
    }
    return res;
  };

  const forward_tensor = (a: Tensor, b: Tensor, operation: TorchFunction | null = null): Tensor => {
    const broadcast_shape = _broadcast_shape(a.shape, b.shape);
    const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
    const padded_b_shape = _pad_shape(b.shape, broadcast_shape);

    const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);

    return new Tensor(
      kernel(
        a.data,
        padded_a_shape,
        b.data,
        padded_b_shape,
        broadcast_shape,
        output_size
      ) as number[],
      { requires_grad: resultRequiresGrad(a, b) },
      { operation: operation, shape: broadcast_shape }
    );
  };

  const result = {
    [opName]: class extends BinaryFunction {
      protected _forward(a: Tensor, b: Tensor): Tensor {
        const rg = resultRequiresGrad(a, b);
        if (rg) {
          this.saved_tensors = [a, b];
        }
        this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
        this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
        return forward_tensor(a, b, rg ? this : null);
      }

      protected _backward(dz: Tensor): void {
        const [a, b] = this.saved_tensors;
        const [aFn, bFn] = this.next_functions;

        backward_operations(a, b, aFn, bFn, dz);
      }
    }
  }[opName];
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
      { requires_grad: resultRequiresGrad(a) },
      { operation: operation, shape: a.shape }
    );
  };

  const result = {
    [opName]: class extends UnaryFunction {
      protected _forward(a: Tensor): Tensor {
        const rg = resultRequiresGrad(a);
        if (rg) {
          this.saved_tensors = [a];
        }
        this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
        return forward_tensor(a, rg ? this : null);
      }

      protected _backward(dz: Tensor): void {
        const [a] = this.saved_tensors;
        const [aFn] = this.next_functions;

        backward_operations(a, aFn, dz);
      }
    }
  }[opName];
  if (opName) {
    registerOperation(opName, result);
  }
  return result;
}

export function ReductionFunctionMixin(
  init_val: number,
  reduce_op: (acc: number, val: number) => number,
  backward_operations: (
    a: Tensor,
    restored_dz: Tensor,
    dim: number | number[],
    keepdim: boolean
  ) => Tensor,
  opName: string | null = null,
  finalize_op?: (acc: number, count: number) => number
): new () => TorchFunction {
  const result = {
    [opName]: class extends TorchFunction {
      protected dim?: number | number[];
      protected keepdim?: boolean;

      protected _forward(a: Tensor, dim?: number | number[], keepdim: boolean = false): Tensor {
        this.dim = dim;
        this.keepdim = keepdim;

        const rg = resultRequiresGrad(a);
        if (rg) {
          this.saved_tensors = [a];
        }
        this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);

        const out_shape = _get_reduction_shape(a.shape, dim, keepdim);
        const out_size = out_shape.reduce((acc, val) => acc * val, 1);

        const res_data = new Array(out_size).fill(init_val);
        const counts = new Array(out_size).fill(0); // Tracked specifically for mean()

        const in_strides = _get_strides(a.shape);
        const out_strides = _get_strides(out_shape);

        const dims = dim === undefined ? [] : Array.isArray(dim) ? dim : [dim];
        const normalized_dims = dims.map(d => (d < 0 ? d + a.shape.length : d));
        const is_full_reduce = dim === undefined;

        // Accumulate
        for (let i = 0; i < a.data.length; i++) {
          const in_coords = _unravel_index(i, in_strides);
          let out_coords: number[];

          if (is_full_reduce) {
            out_coords = keepdim ? in_coords.map(() => 0) : [];
          } else {
            out_coords = [];
            for (let j = 0; j < a.shape.length; j++) {
              if (normalized_dims.includes(j)) {
                if (keepdim) out_coords.push(0); // Collapse dimension to index 0
              } else {
                out_coords.push(in_coords[j]);
              }
            }
          }

          const out_idx = _ravel_index(out_coords, out_strides);
          res_data[out_idx] = reduce_op(res_data[out_idx], a.data[i]);
          counts[out_idx]++;
        }

        // Finalize (e.g., divide by count for mean)
        if (finalize_op) {
          for (let i = 0; i < out_size; i++) {
            res_data[i] = finalize_op(res_data[i], counts[i]);
          }
        }

        return new Tensor(
          res_data,
          { requires_grad: rg },
          { operation: rg ? this : null, shape: out_shape }
        );
      }

      protected _backward(dz: Tensor): void {
        const [a] = this.saved_tensors;
        const [aFn] = this.next_functions;

        let restored_dz = dz;

        const target_shape = _get_reduction_shape(a.shape, this.dim, true);

        if (dz.shape.length !== target_shape.length) {
          restored_dz = dz.reshape(target_shape);
        }

        const expanded_dz = restored_dz.expand(a.shape);
        const grad_a = backward_operations(a, expanded_dz, this.dim, this.keepdim);

        aFn.backward(grad_a);
      }
    }
  }[opName];

  if (opName) {
    registerOperation(opName, result);
  }
  return result;
}
