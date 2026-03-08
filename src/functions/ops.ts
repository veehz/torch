import { Tensor } from '../tensor';
import {
  _broadcast_shape,
  _get_original_index_from_transposed_index,
  _get_original_index,
  _get_original_index_kernel,
  _pad_shape,
  _unbroadcast
} from '../broadcasting';
import { TorchFunction, BinaryFunction, nullOp } from './base';
import * as functional from './functional';
import { registerOperation } from './registry';
import { ones } from '../creation';
import { UnaryFunctionMixin, BinaryFunctionMixin, ReductionFunctionMixin } from './mixin';
import { _get_reduction_shape } from './util';

function unbroadcast(result: Tensor, original_shape: number[]): Tensor {
  const unbroadcasted_result = _unbroadcast(result.shape, original_shape, result.data);
  return new Tensor(unbroadcasted_result, { requires_grad: result.requires_grad }, { shape: original_shape });
}

function broadcast(tensor: Tensor, result_shape: number[]): Tensor {
  return tensor.mul(ones(result_shape));
}

// debug operations

const __Left_index__ = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a_index,
  (a, b, aFn, bFn, dz) => { },
  "__left_index__"
);

const __Right_index__ = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => b_index,
  (a, b, aFn, bFn, dz) => { },
  "__right_index__"
);

// binary pointwise

const Add = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a[a_index] + b[b_index],
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz);
    bFn.backward(dz);
  },
  "add"
);

const Sub = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a[a_index] - b[b_index],
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz);
    bFn.backward(dz.mul(new Tensor(-1)));
  },
  "sub"
);

const Mul = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a[a_index] * b[b_index],
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz.mul(b));
    bFn.backward(dz.mul(a));
  },
  "mul"
);

const Div = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a[a_index] / b[b_index],
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz.div(b));
    bFn.backward(dz.mul(a).mul(new Tensor(-1)).div(b).div(b));
  },
  "div"
);

const Pow = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => Math.pow(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz.mul(b).mul(a.pow(b.sub(new Tensor(1)))));
    bFn.backward(dz.mul(a.pow(b)).mul(a.log()));
  },
  "pow"
);

const Fmod = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a[a_index] % b[b_index],
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz);
  },
  "fmod"
);

const Maximum = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => Math.max(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz.mul(a.ge(b)));
    bFn.backward(dz.mul(b.gt(a)));
  },
  "maximum"
);

const Minimum = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => Math.min(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz.mul(a.le(b)));
    bFn.backward(dz.mul(b.lt(a)));
  },
  "minimum"
);

function _powint_tensor(a: Tensor, n: number, operation: TorchFunction | null = null): Tensor {
  const data = new Array(a.dataLength());
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.pow(a.data[i], n);
  }
  return new Tensor(
    data,
    { requires_grad: a.requires_grad },
    { operation: operation, shape: a.shape }
  );
}

class PowInt extends TorchFunction {
  private n: number;
  protected _forward(a: Tensor, n: number): Tensor {
    if (a.requires_grad) {
      this.saved_tensors = [a];
      this.n = n;
    }

    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _powint_tensor(a, n, a.requires_grad ? this : null);
  }
  protected _backward(dz: Tensor): void {
    const [a] = this.saved_tensors;
    const n = this.n;
    const [aFn] = this.next_functions;

    // backward_operations:
    aFn.backward(dz.mul(n).mul(a.pow(n - 1)));
  }
}
registerOperation("powint", PowInt);

// unary pointwise

const Log = UnaryFunctionMixin(
  (a: number[], a_index: number) => Math.log(a[a_index]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(new Tensor(1).div(a)));
  },
  "log"
);

const Sqrt = UnaryFunctionMixin(
  (a: number[], x: number) => Math.sqrt(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(new Tensor(1).div(a.sqrt()).div(2)));
  },
  "sqrt"
);

const Exp = UnaryFunctionMixin(
  (a: number[], x: number) => Math.exp(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(a.exp())));
  },
  "exp"
);

const Square = UnaryFunctionMixin(
  (a: number[], x: number) => a[x] * a[x],
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(a).mul(new Tensor(2))));
  },
  "square"
);

const Abs = UnaryFunctionMixin(
  (a: number[], x: number) => Math.abs(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(functional.sign(a))));
  },
  "abs"
);

const Sign = UnaryFunctionMixin(
  (a: number[], x: number) => Math.sign(a[x]),
  (a, aFn, dz) => {
    aFn.backward(0);
  },
  "sign"
);

const Neg = UnaryFunctionMixin(
  (a: number[], x: number) => -a[x],
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(new Tensor(-1))));
  },
  "neg"
);

const Reciprocal = UnaryFunctionMixin(
  (a: number[], x: number) => 1 / a[x],
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(a.pow(-2))).neg());
  },
  "reciprocal"
);

class Reshape extends TorchFunction {
  protected _forward(a: Tensor, shape: number[]) {
    const previous_length = a.dataLength();
    const target_length = shape.reduce((acc, val) => acc * val, 1);

    if (previous_length !== target_length) {
      throw new Error('Shape mismatch: ' + a.shape + ' and ' + shape);
    }

    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    if (a.grad_fn) {
      this.next_functions.push(a.grad_fn);
    } else {
      this.next_functions.push(nullOp);
    }

    return new Tensor(
      a.data,
      { requires_grad: a.requires_grad },
      { operation: a.requires_grad ? this : null, shape }
    );
  }
  protected _backward(dz: Tensor) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;

    // backward_operations:
    aFn.backward(dz.reshape(a.shape));
  }
}
registerOperation('reshape', Reshape);

class Squeeze extends TorchFunction {
  protected _forward(a: Tensor, dim?: number) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    if (a.grad_fn) {
      this.next_functions.push(a.grad_fn);
    } else {
      this.next_functions.push(nullOp);
    }

    let shape = [...a.shape];

    if (dim !== undefined) {
      if (dim < 0) {
        dim += a.shape.length;
      }

      // PyTorch only squeezes the specified dimension if its size is exactly 1
      if (shape[dim] === 1) {
        shape.splice(dim, 1);
      }
    } else {
      // If no dim is provided, strip out all dimensions of size 1
      shape = shape.filter((d) => d !== 1);
    }

    return new Tensor(
      a.data,
      { requires_grad: a.requires_grad },
      { operation: a.requires_grad ? this : null, shape }
    );
  }

  protected _backward(dz: Tensor) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;

    // The derivative of squeeze is just reshaping the gradient 
    // back to the original unsqueezed shape.
    aFn.backward(dz.reshape(a.shape));
  }
}
registerOperation('squeeze', Squeeze);

class Unsqueeze extends TorchFunction {
  protected _forward(a: Tensor, dim: number) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    if (a.grad_fn) {
      this.next_functions.push(a.grad_fn);
    } else {
      this.next_functions.push(nullOp);
    }

    if (dim < 0) {
      dim += a.shape.length + 1;
    }

    const shape = [...a.shape];
    shape.splice(dim, 0, 1);

    return new Tensor(
      a.data,
      { requires_grad: a.requires_grad },
      { operation: a.requires_grad ? this : null, shape }
    );
  }
  protected _backward(dz: Tensor) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;

    // backward_operations:
    aFn.backward(dz.reshape(a.shape));
  }
}
registerOperation('unsqueeze', Unsqueeze);

class Expand extends TorchFunction {
  protected _forward(a: Tensor, expanded_shape: number[]): Tensor {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    if (a.grad_fn) {
      this.next_functions.push(a.grad_fn);
    } else {
      this.next_functions.push(nullOp);
    }
  
    const offset = expanded_shape.length - a.shape.length;
    const target_shape = expanded_shape.map((dim, i) => {
      if (dim === -1) {
        const orig_i = i - offset;
        return orig_i >= 0 ? a.shape[orig_i] : 1;
      }
      return dim;
    });

    // Steal data from just broadcasting
    const outData = broadcast(a, target_shape).data;

    return new Tensor(
      outData,
      { requires_grad: a.requires_grad },
      { operation: a.requires_grad ? this : null, shape: target_shape }
    );
  }

  protected _backward(dz: Tensor): void {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;

    // Route the collapsed gradient upstream
    aFn.backward(unbroadcast(dz, a.shape));
  }
}
registerOperation('expand', Expand)

// trigonometric

const Sin = UnaryFunctionMixin(
  (a: number[], x: number) => Math.sin(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(a.cos())));
  },
  "sin"
);

const Cos = UnaryFunctionMixin(
  (a: number[], x: number) => Math.cos(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(a.sin().neg())));
  },
  "cos"
);

const Tan = UnaryFunctionMixin(
  (a: number[], x: number) => Math.tan(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(dz.mul(a.cos().pow(-2))));
  },
  "tan"
);

// reduction

export const Sum = ReductionFunctionMixin(
  0,
  (acc, val) => acc + val,
  (a, expanded_dz) => expanded_dz,
  'sum'
);

export const Mean = ReductionFunctionMixin(
  0,
  (acc, val) => acc + val,
  (a, expanded_dz, dim) => {
    const target_shape = _get_reduction_shape(a.shape, dim, false);
    const out_size = target_shape.length > 0 ? target_shape.reduce((acc, v) => acc * v, 1) : 1;
    const N = a.dataLength() / out_size;

    return expanded_dz.mul(new Tensor([1 / N])); 
  },
  'mean',
  (acc, count) => acc / count
);

export const Max = ReductionFunctionMixin(
  -Infinity,
  (acc, val) => Math.max(acc, val),
  (a, expanded_dz, dim) => {
    const max_tensor = a.max(dim, true);
    const max_expanded = max_tensor.expand(a.shape);
    const mask = a.eq(max_expanded).detach();

    return expanded_dz.mul(mask);
  },
  'max'
);

export const Min = ReductionFunctionMixin(
  Infinity,
  (acc, val) => Math.min(acc, val),
  (a, expanded_dz, dim) => {
    const min_tensor = a.min(dim, true);
    const min_expanded = min_tensor.expand(a.shape);
    const mask = a.eq(min_expanded).detach();

    return expanded_dz.mul(mask);
  },
  'min'
);

// linalg

function _transpose_tensor(
  a: Tensor,
  dim0: number,
  dim1: number,
  operation: TorchFunction | null = null
): Tensor {
  if (a.shape.length + dim0 < 0 || a.shape.length + dim1 < 0) {
    throw new Error(`Transpose: Dimension out of range (${dim0} and ${dim1})`);
  }
  dim0 = dim0 < 0 ? a.shape.length + dim0 : dim0;
  dim1 = dim1 < 0 ? a.shape.length + dim1 : dim1;

  const output_shape = [...a.shape];
  [output_shape[dim0], output_shape[dim1]] = [output_shape[dim1], output_shape[dim0]];
  const size = a.dataLength();
  const data = new Array(size);

  const a_strides = new Array(a.shape.length);
  const out_strides = new Array(output_shape.length);
  for (let i = a.shape.length - 1, s = 1; i >= 0; i--) {
    a_strides[i] = s;
    s *= a.shape[i];
  }
  for (let i = output_shape.length - 1, s = 1; i >= 0; i--) {
    out_strides[i] = s;
    s *= output_shape[i];
  }

  for (let i = 0; i < size; i++) {
    let idx = i;
    let input_idx = 0;
    for (let d = 0; d < output_shape.length; d++) {
      const stride = out_strides[d];
      const coord = Math.floor(idx / stride);
      idx %= stride;

      let input_d = d;
      if (d === dim0) input_d = dim1;
      else if (d === dim1) input_d = dim0;

      input_idx += coord * a_strides[input_d];
    }
    data[i] = a.data[input_idx];
  }

  return new Tensor(
    data,
    { requires_grad: a.requires_grad },
    { operation: operation, shape: output_shape }
  );
}
class Transpose extends TorchFunction {
  private dim0: number;
  private dim1: number;
  protected _forward(a: Tensor, dim0: number, dim1: number): Tensor {
    if (a.requires_grad) {
      this.saved_tensors = [a];
      this.dim0 = dim0;
      this.dim1 = dim1;
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _transpose_tensor(a, dim0, dim1, this);
  }
  protected _backward(dz: Tensor): void {
    const [a] = this.saved_tensors;
    const dim0 = this.dim0;
    const dim1 = this.dim1;
    const [aFn] = this.next_functions;

    // backward_operations:
    aFn.backward(dz.transpose(dim0, dim1));
  }
}
registerOperation('transpose', Transpose);

function _matmul_tensor(a: Tensor, b: Tensor, operation: TorchFunction | null = null): [Tensor, number[]] {
  if (a.shape.length == 1 && b.shape.length == 1) {
    return [a.mul(b).sum(), []];
  }

  const a_1d = a.shape.length == 1;
  const b_1d = b.shape.length == 1;

  const a_shape = a_1d ? [1, a.shape[0]] : a.shape;
  const b_shape = b_1d ? [b.shape[0], 1] : b.shape;

  if (a_shape[a_shape.length - 1] != b_shape[b_shape.length - 2]) {
    throw new Error('Shape mismatch: ' + a.shape + ' and ' + b.shape);
  }

  const broadcast_shape = _broadcast_shape(a_shape.slice(0, -2), b_shape.slice(0, -2)).concat([
    a_shape[a_shape.length - 2],
    b_shape[b_shape.length - 1]
  ]);

  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  const data = new Array(output_size).fill(0);

  const padded_a_shape = _pad_shape(a_shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b_shape, broadcast_shape);

  const dim_M = broadcast_shape[broadcast_shape.length - 2];
  const dim_N = broadcast_shape[broadcast_shape.length - 1];
  const dim_K = a_shape[a_shape.length - 1]; // or b_shape[b_shape.length - 2]

  for (let i = 0; i < output_size; i++) {
    const mn_idx = i % (dim_M * dim_N);
    const m = Math.floor(mn_idx / dim_N);
    const n = mn_idx % dim_N;

    let base_a = _get_original_index(padded_a_shape, broadcast_shape, i - n);
    let base_b = _get_original_index(padded_b_shape, broadcast_shape, i - m * dim_N);

    let sum = 0;
    for (let k = 0; k < dim_K; k++) {
      sum += a.data[base_a + k] * b.data[base_b + k * dim_N];
    }
    data[i] = sum;
  }

  let shape_after_removing_extra_dims = [...broadcast_shape];

  if (a_1d) {
    shape_after_removing_extra_dims = shape_after_removing_extra_dims
      .slice(0, -2)
      .concat([broadcast_shape[broadcast_shape.length - 1]]);
  }

  if (b_1d) {
    shape_after_removing_extra_dims = shape_after_removing_extra_dims.slice(0, -1);
  }

  return [new Tensor(
    data,
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation: operation, shape: shape_after_removing_extra_dims }
  ), shape_after_removing_extra_dims];
}

class Matmul extends BinaryFunction {
  private shape: number[];

  protected _forward(a: Tensor, b: Tensor): Tensor {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    const result = _matmul_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
    this.shape = result[1];
    return result[0];
  }
  protected _backward(dz: Tensor): void {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;

    // 1. 1D x 1D (Dot Product)
    if (a.shape.length === 1 && b.shape.length === 1) {
      aFn.backward(dz.mul(b));
      bFn.backward(dz.mul(a));
      return;
    }

    // 2. 1D x ND
    if (a.shape.length === 1) {
      const dz1 = dz.unsqueeze(-2);
      const a1 = a.unsqueeze(-2);

      let da = dz1.matmul(b.transpose(-2, -1));
      let db = a1.transpose(-2, -1).matmul(dz1);

      da = da.squeeze(-2);
      db = unbroadcast(db, b.shape);

      aFn.backward(da);
      bFn.backward(db);
      return;
    }

    // 3. ND x 1D
    if (b.shape.length === 1) {
      const dz1 = dz.unsqueeze(-1);
      const b1 = b.unsqueeze(-1);

      let da = dz1.matmul(b1.transpose(-2, -1));
      let db = a.transpose(-2, -1).matmul(dz1);

      da = unbroadcast(da, a.shape);
      db = db.squeeze(-1);

      aFn.backward(da);
      bFn.backward(db);
      return;
    }

    // 4. ND x ND (Batched or Standard)
    let da = dz.matmul(b.transpose(-2, -1));
    let db = a.transpose(-2, -1).matmul(dz);

    da = unbroadcast(da, a.shape);
    db = unbroadcast(db, b.shape);

    aFn.backward(da);
    bFn.backward(db);
  }
}
registerOperation('matmul', Matmul);

// comparison

const Lt = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] < b[b_index]) ? 1 : 0,
  (a, b, aFn, bFn) => { },
  "lt"
);

const Gt = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] > b[b_index]) ? 1 : 0,
  (a, b, aFn, bFn) => { },
  "gt"
);

const Le = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] <= b[b_index]) ? 1 : 0,
  (a, b, aFn, bFn) => { },
  "le"
);

const Ge = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] >= b[b_index]) ? 1 : 0,
  (a, b, aFn, bFn) => { },
  "ge"
);

const Eq = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] == b[b_index]) ? 1 : 0,
  (a, b, aFn, bFn) => { },
  "eq"
);

const Ne = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] != b[b_index]) ? 1 : 0,
  (a, b, aFn, bFn) => { },
  "ne"
);
