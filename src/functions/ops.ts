import { Tensor } from '../tensor';
import {
  _broadcast_shape,
  _get_original_index,
  _pad_shape,
  _unbroadcast
} from '../broadcasting';
import { TorchFunction, BinaryFunction, nullOp, resultRequiresGrad } from './base';
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
  (a: number[], b: number[], a_index: number, _b_index: number) => a_index,
  () => { },
  "__left_index__"
);

const __Right_index__ = BinaryFunctionMixin(
  (a: number[], b: number[], _a_index: number, b_index: number) => b_index,
  () => { },
  "__right_index__"
);

// binary pointwise

const Add = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a[a_index] + b[b_index],
  (_a, _b, aFn, bFn, dz) => {
    aFn.backward(dz);
    bFn.backward(dz);
  },
  "add"
);

const Sub = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a[a_index] - b[b_index],
  (_a, _b, aFn, bFn, dz) => {
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

function _where(mask: Tensor, x: Tensor, fallback: Tensor | number): Tensor {
  const fb = typeof fallback === 'number' ? fallback : null;
  const maskData = mask.data;
  const xData = x.data;
  const fbData = fb === null ? (fallback as Tensor).data : null;
  const data = new Array(x.dataLength());
  for (let i = 0; i < data.length; i++) {
    data[i] = maskData[i] ? xData[i] : (fb !== null ? fb : fbData![i]);
  }
  return new Tensor(data, {}, { shape: x.shape });
}

const Pow = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => Math.pow(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    const ga = dz.mul(b).mul(a.pow(b.sub(new Tensor(1))));
    const gb = dz.mul(a.pow(b)).mul(a.log());
    // When a==0, grads can produce NaN/Inf (from 0*Inf or log(0)); replace with 0
    aFn.backward(_where(a.ne(0), ga, ga.nan_to_num()));
    bFn.backward(_where(a.ne(0), gb, 0));
  },
  "pow"
);

const Fmod = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => a[a_index] % b[b_index],
  (_a, _b, aFn, _bFn, dz) => {
    aFn.backward(dz);
  },
  "fmod"
);

const Maximum = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => Math.max(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    // When a == b, PyTorch splits gradient 0.5 each
    const eq_mask = a.eq(b);
    const a_mask = a.gt(b).add(eq_mask.mul(new Tensor(0.5)));
    const b_mask = b.gt(a).add(eq_mask.mul(new Tensor(0.5)));
    aFn.backward(dz.mul(a_mask));
    bFn.backward(dz.mul(b_mask));
  },
  "maximum"
);

const Minimum = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => Math.min(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    // When a == b, PyTorch splits gradient 0.5 each
    const eq_mask = a.eq(b);
    const a_mask = a.lt(b).add(eq_mask.mul(new Tensor(0.5)));
    const b_mask = b.lt(a).add(eq_mask.mul(new Tensor(0.5)));
    aFn.backward(dz.mul(a_mask));
    bFn.backward(dz.mul(b_mask));
  },
  "minimum"
);

function _powint_tensor(a: Tensor, n: number, operation: TorchFunction | null = null): Tensor {
  const aData = a.data;
  const data = new Array(a.dataLength());
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.pow(aData[i], n);
  }
  return new Tensor(
    data,
    { requires_grad: resultRequiresGrad(a) },
    { operation: operation, shape: a.shape }
  );
}

class PowInt extends TorchFunction {
  private n: number;
  protected _forward(a: Tensor, n: number): Tensor {
    const rg = resultRequiresGrad(a);
    if (rg) {
      this.saved_tensors = [a];
      this.n = n;
    }

    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _powint_tensor(a, n, rg ? this : null);
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
    aFn.backward(dz.mul(a.exp()));
  },
  "exp"
);

const Square = UnaryFunctionMixin(
  (a: number[], x: number) => a[x] * a[x],
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a).mul(new Tensor(2)));
  },
  "square"
);

const Abs = UnaryFunctionMixin(
  (a: number[], x: number) => Math.abs(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(functional.sign(a)));
  },
  "abs"
);

const Sign = UnaryFunctionMixin(
  (a: number[], x: number) => Math.sign(a[x]),
  (_a, aFn) => {
    aFn.backward(0);
  },
  "sign"
);

const Neg = UnaryFunctionMixin(
  (a: number[], x: number) => -a[x],
  (_a, aFn, dz) => {
    aFn.backward(dz.mul(new Tensor(-1)));
  },
  "neg"
);

const Reciprocal = UnaryFunctionMixin(
  (a: number[], x: number) => 1 / a[x],
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.pow(-2)).neg());
  },
  "reciprocal"
);

const NanToNum = UnaryFunctionMixin(
  (a: number[], x: number) => {
    const v = a[x];
    if (Number.isNaN(v)) return 0;
    if (v === Infinity) return 3.4028235e+38;
    if (v === -Infinity) return -3.4028235e+38;
    return v;
  },
  (a, aFn, dz) => {
    aFn.backward(dz);
  },
  "nan_to_num"
);

class Reshape extends TorchFunction {
  protected _forward(a: Tensor, shape: number[]) {
    const previous_length = a.dataLength();

    const negIdx = shape.indexOf(-1);
    if (negIdx !== -1) {
      if (shape.indexOf(-1, negIdx + 1) !== -1) {
        throw new Error('Only one -1 is allowed in reshape shape');
      }
      const known = shape.reduce((acc, val, i) => i === negIdx ? acc : acc * val, 1);
      if (previous_length % known !== 0) {
        throw new Error('Shape mismatch: cannot infer -1 dimension for shape ' + a.shape + ' -> ' + shape);
      }
      shape = shape.slice();
      shape[negIdx] = previous_length / known;
    }

    const target_length = shape.reduce((acc, val) => acc * val, 1);

    if (previous_length !== target_length) {
      throw new Error('Shape mismatch: ' + a.shape + ' and ' + shape);
    }

    const rg = resultRequiresGrad(a);
    if (rg) {
      this.saved_tensors = [a];
    }
    if (a.grad_fn) {
      this.next_functions.push(a.grad_fn);
    } else {
      this.next_functions.push(nullOp);
    }

    return new Tensor(
      a.data,
      { requires_grad: rg },
      { operation: rg ? this : null, shape }
    );
  }
  protected _backward(dz: Tensor) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;

    aFn.backward(dz.reshape(a.shape));
  }
}
registerOperation('reshape', Reshape);

class Flatten extends TorchFunction {
  protected _forward(a: Tensor, start_dim: number = 0, end_dim: number = -1) {
    const ndim = a.shape.length;
    const sd = start_dim < 0 ? start_dim + ndim : start_dim;
    const ed = end_dim < 0 ? end_dim + ndim : end_dim;
    const newShape = [
      ...a.shape.slice(0, sd),
      a.shape.slice(sd, ed + 1).reduce((acc, val) => acc * val, 1),
      ...a.shape.slice(ed + 1),
    ];

    const rg = resultRequiresGrad(a);
    if (rg) {
      this.saved_tensors = [a];
    }
    if (a.grad_fn) {
      this.next_functions.push(a.grad_fn);
    } else {
      this.next_functions.push(nullOp);
    }

    return new Tensor(
      a.data,
      { requires_grad: rg },
      { operation: rg ? this : null, shape: newShape }
    );
  }
  protected _backward(dz: Tensor) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;

    aFn.backward(dz.reshape(a.shape));
  }
}
registerOperation('flatten', Flatten);

class Squeeze extends TorchFunction {
  protected _forward(a: Tensor, dim?: number) {
    const rg = resultRequiresGrad(a);
    if (rg) {
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
      { requires_grad: rg },
      { operation: rg ? this : null, shape }
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
    const rg = resultRequiresGrad(a);
    if (rg) {
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
      { requires_grad: rg },
      { operation: rg ? this : null, shape }
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
    const rg = resultRequiresGrad(a);
    if (rg) {
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
      { requires_grad: rg },
      { operation: rg ? this : null, shape: target_shape }
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
    aFn.backward(dz.mul(a.cos()));
  },
  "sin"
);

const Cos = UnaryFunctionMixin(
  (a: number[], x: number) => Math.cos(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.sin().neg()));
  },
  "cos"
);

const Tan = UnaryFunctionMixin(
  (a: number[], x: number) => Math.tan(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.cos().pow(-2)));
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
  const aData = a.data;

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
    data[i] = aData[input_idx];
  }

  return new Tensor(
    data,
    { requires_grad: resultRequiresGrad(a) },
    { operation: operation, shape: output_shape }
  );
}
class Transpose extends TorchFunction {
  private dim0: number;
  private dim1: number;
  protected _forward(a: Tensor, dim0: number, dim1: number): Tensor {
    const rg = resultRequiresGrad(a);
    if (rg) {
      this.saved_tensors = [a];
      this.dim0 = dim0;
      this.dim1 = dim1;
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _transpose_tensor(a, dim0, dim1, rg ? this : null);
  }
  protected _backward(dz: Tensor): void {
    // const [a] = this.saved_tensors;
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

  const aData = a.data;
  const bData = b.data;
  for (let i = 0; i < output_size; i++) {
    const mn_idx = i % (dim_M * dim_N);
    const m = Math.floor(mn_idx / dim_N);
    const n = mn_idx % dim_N;

    const base_a = _get_original_index(padded_a_shape, broadcast_shape, i - n);
    const base_b = _get_original_index(padded_b_shape, broadcast_shape, i - m * dim_N);

    let sum = 0;
    for (let k = 0; k < dim_K; k++) {
      sum += aData[base_a + k] * bData[base_b + k * dim_N];
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
    { requires_grad: resultRequiresGrad(a, b) },
    { operation: operation, shape: shape_after_removing_extra_dims }
  ), shape_after_removing_extra_dims];
}

class Matmul extends BinaryFunction {
  private shape: number[];

  protected _forward(a: Tensor, b: Tensor): Tensor {
    const rg = resultRequiresGrad(a, b);
    if (rg) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    const result = _matmul_tensor(a, b, rg ? this : null);
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

function _convNd_forward(
  input: Tensor,
  weight: Tensor,
  bias: Tensor | null,
  stride: number | number[],
  padding: number | number[],
  dilation: number | number[],
  groups: number,
  dims: number
): Tensor {
  const stride_arr = typeof stride === 'number' ? new Array(dims).fill(stride) : stride;
  const padding_arr = typeof padding === 'number' ? new Array(dims).fill(padding) : padding;
  const dilation_arr = typeof dilation === 'number' ? new Array(dims).fill(dilation) : dilation;

  const batch_size = input.shape[0];
  const in_channels = input.shape[1];
  const out_channels = weight.shape[0];
  const in_dims = input.shape.slice(2);
  const kernel_dims = weight.shape.slice(2);

  if (in_channels !== weight.shape[1] * groups) {
    throw new Error(`in_channels (${in_channels}) must be divisible by groups (${groups}) and match weight.shape[1] * groups (${weight.shape[1] * groups})`);
  }

  const out_dims = in_dims.map((in_dim, i) => {
    return Math.floor((in_dim + 2 * padding_arr[i] - dilation_arr[i] * (kernel_dims[i] - 1) - 1) / stride_arr[i] + 1);
  });

  const output_shape = [batch_size, out_channels, ...out_dims];
  const output_size = output_shape.reduce((a, b) => a * b, 1);
  const output_data = new Array(output_size).fill(0);

  const get_strides = (shape: number[]) => {
    const strides = new Array(shape.length);
    let s = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = s;
      s *= shape[i];
    }
    return strides;
  };

  const in_strides = get_strides(input.shape);
  const w_strides = get_strides(weight.shape);
  const out_strides = get_strides(output_shape);
  const in_channels_per_group = in_channels / groups;
  const out_channels_per_group = out_channels / groups;
  const inputData = input.data;
  const weightData = weight.data;
  const biasData = bias ? bias.data : null;

  for (let b = 0; b < batch_size; b++) {
    for (let g = 0; g < groups; g++) {
      for (let oc_g = 0; oc_g < out_channels_per_group; oc_g++) {
        const oc = g * out_channels_per_group + oc_g;

        // Iterate over output spatial dimensions
        const out_spatial_size = out_dims.reduce((a, b) => a * b, 1);
        for (let os_idx = 0; os_idx < out_spatial_size; os_idx++) {

          // Decode output spatial index
          const os_coords = new Array(dims);
          let temp_os = os_idx;
          for (let d = dims - 1; d >= 0; d--) {
            os_coords[d] = temp_os % out_dims[d];
            temp_os = Math.floor(temp_os / out_dims[d]);
          }

          let sum = biasData ? biasData[oc] : 0;

          // Iterate over kernel spatial dimensions and in_channels
          for (let ic_g = 0; ic_g < in_channels_per_group; ic_g++) {
            const ic = g * in_channels_per_group + ic_g;

            const kernel_spatial_size = kernel_dims.reduce((a, b) => a * b, 1);
            for (let ks_idx = 0; ks_idx < kernel_spatial_size; ks_idx++) {
              // Decode kernel spatial index
              const ks_coords = new Array(dims);
              let temp_ks = ks_idx;
              for (let d = dims - 1; d >= 0; d--) {
                ks_coords[d] = temp_ks % kernel_dims[d];
                temp_ks = Math.floor(temp_ks / kernel_dims[d]);
              }

              // Calculate input spatial coordinates
              let is_valid = true;
              const is_coords = new Array(dims);
              for (let d = 0; d < dims; d++) {
                const in_coord = os_coords[d] * stride_arr[d] + ks_coords[d] * dilation_arr[d] - padding_arr[d];
                if (in_coord < 0 || in_coord >= in_dims[d]) {
                  is_valid = false;
                  break;
                }
                is_coords[d] = in_coord;
              }

              if (is_valid) {
                // Calculate flattened indices
                let in_flat_idx = b * in_strides[0] + ic * in_strides[1];
                for (let d = 0; d < dims; d++) in_flat_idx += is_coords[d] * in_strides[d + 2];

                let w_flat_idx = oc * w_strides[0] + ic_g * w_strides[1];
                for (let d = 0; d < dims; d++) w_flat_idx += ks_coords[d] * w_strides[d + 2];

                sum += inputData[in_flat_idx] * weightData[w_flat_idx];
              }
            }
          }

          // Calculate output flattened index
          let out_flat_idx = b * out_strides[0] + oc * out_strides[1];
          for (let d = 0; d < dims; d++) out_flat_idx += os_coords[d] * out_strides[d + 2];

          output_data[out_flat_idx] = sum;
        }
      }
    }
  }

  return new Tensor(output_data, { requires_grad: false }, { shape: output_shape });
}

function _convNd_backward(
  dz: Tensor,
  input: Tensor,
  weight: Tensor,
  bias: Tensor | null,
  stride: number | number[],
  padding: number | number[],
  dilation: number | number[],
  groups: number,
  dims: number,
  input_requires_grad: boolean,
  weight_requires_grad: boolean
): [Tensor | null, Tensor | null, Tensor | null] {
  const stride_arr = typeof stride === 'number' ? new Array(dims).fill(stride) : stride;
  const padding_arr = typeof padding === 'number' ? new Array(dims).fill(padding) : padding;
  const dilation_arr = typeof dilation === 'number' ? new Array(dims).fill(dilation) : dilation;

  const batch_size = input.shape[0];
  const in_channels = input.shape[1];
  const out_channels = weight.shape[0];
  const in_dims = input.shape.slice(2);
  const kernel_dims = weight.shape.slice(2);
  const out_dims = dz.shape.slice(2);

  const get_strides = (shape: number[]) => {
    const strides = new Array(shape.length);
    let s = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = s;
      s *= shape[i];
    }
    return strides;
  };

  const in_strides = get_strides(input.shape);
  const w_strides = get_strides(weight.shape);
  const dz_strides = get_strides(dz.shape);
  const dzData = dz.data;
  const weightDataBwd = weight.data;
  const inputDataBwd = input.data;

  let dInput: Tensor | null = null;
  let dWeight: Tensor | null = null;
  let dBias: Tensor | null = null;

  let dInput_data: number[] | null = null;
  let dWeight_data: number[] | null = null;

  if (input_requires_grad) {
    dInput_data = new Array(input.dataLength()).fill(0);
  }
  if (weight_requires_grad) {
    dWeight_data = new Array(weight.dataLength()).fill(0);
  }

  const in_channels_per_group = in_channels / groups;
  const out_channels_per_group = out_channels / groups;

  for (let b = 0; b < batch_size; b++) {
    for (let g = 0; g < groups; g++) {
      for (let oc_g = 0; oc_g < out_channels_per_group; oc_g++) {
        const oc = g * out_channels_per_group + oc_g;

        const out_spatial_size = out_dims.reduce((a, b) => a * b, 1);
        for (let os_idx = 0; os_idx < out_spatial_size; os_idx++) {

          const os_coords = new Array(dims);
          let temp_os = os_idx;
          for (let d = dims - 1; d >= 0; d--) {
            os_coords[d] = temp_os % out_dims[d];
            temp_os = Math.floor(temp_os / out_dims[d]);
          }

          let dz_flat_idx = b * dz_strides[0] + oc * dz_strides[1];
          for (let d = 0; d < dims; d++) dz_flat_idx += os_coords[d] * dz_strides[d + 2];
          const dz_val = dzData[dz_flat_idx];

          for (let ic_g = 0; ic_g < in_channels_per_group; ic_g++) {
            const ic = g * in_channels_per_group + ic_g;

            const kernel_spatial_size = kernel_dims.reduce((a, b) => a * b, 1);
            for (let ks_idx = 0; ks_idx < kernel_spatial_size; ks_idx++) {
              const ks_coords = new Array(dims);
              let temp_ks = ks_idx;
              for (let d = dims - 1; d >= 0; d--) {
                ks_coords[d] = temp_ks % kernel_dims[d];
                temp_ks = Math.floor(temp_ks / kernel_dims[d]);
              }

              let is_valid = true;
              const is_coords = new Array(dims);
              for (let d = 0; d < dims; d++) {
                const in_coord = os_coords[d] * stride_arr[d] + ks_coords[d] * dilation_arr[d] - padding_arr[d];
                if (in_coord < 0 || in_coord >= in_dims[d]) {
                  is_valid = false;
                  break;
                }
                is_coords[d] = in_coord;
              }

              if (is_valid) {
                let in_flat_idx = b * in_strides[0] + ic * in_strides[1];
                for (let d = 0; d < dims; d++) in_flat_idx += is_coords[d] * in_strides[d + 2];

                let w_flat_idx = oc * w_strides[0] + ic_g * w_strides[1];
                for (let d = 0; d < dims; d++) w_flat_idx += ks_coords[d] * w_strides[d + 2];

                if (input_requires_grad) {
                  dInput_data![in_flat_idx] += dz_val * weightDataBwd[w_flat_idx];
                }
                if (weight_requires_grad) {
                  dWeight_data![w_flat_idx] += dz_val * inputDataBwd[in_flat_idx];
                }
              }
            }
          }
        }
      }
    }
  }

  if (input_requires_grad) dInput = new Tensor(dInput_data!, { requires_grad: false }, { shape: input.shape });
  if (weight_requires_grad) dWeight = new Tensor(dWeight_data!, { requires_grad: false }, { shape: weight.shape });
  if (bias && bias.requires_grad) {
    const sum_dims = [0];
    for (let d = 2; d < dz.shape.length; d++) sum_dims.push(d);
    dBias = dz.sum(sum_dims);
  }

  return [dInput, dWeight, dBias];
}

class Conv1dOp extends TorchFunction {
  private stride: number | number[];
  private padding: number | number[];
  private dilation: number | number[];
  private groups: number;

  protected _forward(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    groups: number = 1
  ): Tensor {
    const rg = resultRequiresGrad(input, weight, ...(bias ? [bias] : []));
    if (rg) {
      this.saved_tensors = [input, weight];
      if (bias) this.saved_tensors.push(bias);
    }
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);
    this.next_functions.push(weight.grad_fn ? weight.grad_fn : nullOp);
    if (bias) this.next_functions.push(bias.grad_fn ? bias.grad_fn : nullOp);

    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;

    const res = _convNd_forward(input, weight, bias, stride, padding, dilation, groups, 1);
    res.requires_grad = rg;
    res.grad_fn = rg ? this : null;
    return res;
  }

  protected _backward(dz: Tensor): void {
    const input = this.saved_tensors[0];
    const weight = this.saved_tensors[1];
    const bias = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null;
    const [inputFn, weightFn, biasFn] = this.next_functions;

    const [dInput, dWeight, dBias] = _convNd_backward(
      dz, input, weight, bias, this.stride, this.padding, this.dilation, this.groups, 1,
      input.requires_grad, weight.requires_grad
    );

    if (input.requires_grad) inputFn.backward(dInput);
    if (weight.requires_grad) weightFn.backward(dWeight);
    if (bias && bias.requires_grad) biasFn.backward(dBias);
  }
}
registerOperation('conv1d', Conv1dOp);

class Conv2dOp extends TorchFunction {
  private stride: number | number[];
  private padding: number | number[];
  private dilation: number | number[];
  private groups: number;

  protected _forward(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    groups: number = 1
  ): Tensor {
    const rg = resultRequiresGrad(input, weight, ...(bias ? [bias] : []));
    if (rg) {
      this.saved_tensors = [input, weight];
      if (bias) this.saved_tensors.push(bias);
    }
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);
    this.next_functions.push(weight.grad_fn ? weight.grad_fn : nullOp);
    if (bias) this.next_functions.push(bias.grad_fn ? bias.grad_fn : nullOp);

    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;

    const res = _convNd_forward(input, weight, bias, stride, padding, dilation, groups, 2);
    res.requires_grad = rg;
    res.grad_fn = rg ? this : null;
    return res;
  }

  protected _backward(dz: Tensor): void {
    const input = this.saved_tensors[0];
    const weight = this.saved_tensors[1];
    const bias = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null;
    const [inputFn, weightFn, biasFn] = this.next_functions;

    const [dInput, dWeight, dBias] = _convNd_backward(
      dz, input, weight, bias, this.stride, this.padding, this.dilation, this.groups, 2,
      input.requires_grad, weight.requires_grad
    );

    if (input.requires_grad) inputFn.backward(dInput);
    if (weight.requires_grad) weightFn.backward(dWeight);
    if (bias && bias.requires_grad) biasFn.backward(dBias);
  }
}
registerOperation('conv2d', Conv2dOp);

class Conv3dOp extends TorchFunction {
  private stride: number | number[];
  private padding: number | number[];
  private dilation: number | number[];
  private groups: number;

  protected _forward(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    groups: number = 1
  ): Tensor {
    const rg = resultRequiresGrad(input, weight, ...(bias ? [bias] : []));
    if (rg) {
      this.saved_tensors = [input, weight];
      if (bias) this.saved_tensors.push(bias);
    }
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);
    this.next_functions.push(weight.grad_fn ? weight.grad_fn : nullOp);
    if (bias) this.next_functions.push(bias.grad_fn ? bias.grad_fn : nullOp);

    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;

    const res = _convNd_forward(input, weight, bias, stride, padding, dilation, groups, 3);
    res.requires_grad = rg;
    res.grad_fn = rg ? this : null;
    return res;
  }

  protected _backward(dz: Tensor): void {
    const input = this.saved_tensors[0];
    const weight = this.saved_tensors[1];
    const bias = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null;
    const [inputFn, weightFn, biasFn] = this.next_functions;

    const [dInput, dWeight, dBias] = _convNd_backward(
      dz, input, weight, bias, this.stride, this.padding, this.dilation, this.groups, 3,
      input.requires_grad, weight.requires_grad
    );

    if (input.requires_grad) inputFn.backward(dInput);
    if (weight.requires_grad) weightFn.backward(dWeight);
    if (bias && bias.requires_grad) biasFn.backward(dBias);
  }
}
registerOperation('conv3d', Conv3dOp);


// comparison

const Lt = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] < b[b_index]) ? 1 : 0,
  () => { },
  "lt"
);

const Gt = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] > b[b_index]) ? 1 : 0,
  () => { },
  "gt"
);

const Le = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] <= b[b_index]) ? 1 : 0,
  () => { },
  "le"
);

const Ge = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] >= b[b_index]) ? 1 : 0,
  () => { },
  "ge"
);

const Eq = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] == b[b_index]) ? 1 : 0,
  () => { },
  "eq"
);

const Ne = BinaryFunctionMixin(
  (a: number[], b: number[], a_index: number, b_index: number) => (a[a_index] != b[b_index]) ? 1 : 0,
  () => { },
  "ne"
);

class Cat extends TorchFunction {
  private _dim: number;
  private _input_sizes: number[];

  protected _forward(tensors: Tensor[], dim: number = 0): Tensor {
    if (tensors.length === 0) {
      throw new Error('torch.cat: expected a non-empty list of Tensors');
    }

    const ndim = tensors[0].shape.length;
    if (ndim === 0) {
      throw new Error('torch.cat: zero-dimensional tensors cannot be concatenated');
    }

    if (dim < 0) dim = dim + ndim;
    if (dim < 0 || dim >= ndim) {
      throw new Error(`torch.cat: dimension out of range`);
    }

    for (let i = 1; i < tensors.length; i++) {
      if (tensors[i].shape.length !== ndim) {
        throw new Error('torch.cat: all tensors must have the same number of dimensions');
      }
      for (let d = 0; d < ndim; d++) {
        if (d !== dim && tensors[i].shape[d] !== tensors[0].shape[d]) {
          throw new Error(`torch.cat: all tensors must have the same shape, except in dimension ${dim}`);
        }
      }
    }

    const rg = resultRequiresGrad(...tensors);
    if (rg) {
      this.saved_tensors = [...tensors];
      this._dim = dim;
      this._input_sizes = tensors.map(t => t.shape[dim]);
    }

    for (const t of tensors) {
      this.next_functions.push(t.grad_fn ? t.grad_fn : nullOp);
    }

    const result_shape = [...tensors[0].shape];
    result_shape[dim] = tensors.reduce((sum, t) => sum + t.shape[dim], 0);

    const inner_size = result_shape.slice(dim + 1).reduce((a, b) => a * b, 1);
    const outer_size = result_shape.slice(0, dim).reduce((a, b) => a * b, 1);
    const total_size = result_shape.reduce((a, b) => a * b, 1);

    const data = new Array(total_size);
    let out_idx = 0;
    for (let o = 0; o < outer_size; o++) {
      for (const t of tensors) {
        const block_size = t.shape[dim] * inner_size;
        const tData = t.data;
        const src_base = o * block_size;
        for (let i = 0; i < block_size; i++) {
          data[out_idx++] = tData[src_base + i];
        }
      }
    }

    return new Tensor(data, { requires_grad: rg }, { operation: rg ? this : null, shape: result_shape });
  }

  protected _backward(dz: Tensor): void {
    const tensors = this.saved_tensors;
    const dim = this._dim;
    const input_sizes = this._input_sizes;

    const inner_size = dz.shape.slice(dim + 1).reduce((a, b) => a * b, 1);
    const outer_size = dz.shape.slice(0, dim).reduce((a, b) => a * b, 1);
    const dzData = dz.data;

    const grad_datas: number[][] = tensors.map(t => new Array(t.dataLength()));

    let dz_idx = 0;
    for (let o = 0; o < outer_size; o++) {
      for (let i = 0; i < tensors.length; i++) {
        const block_size = input_sizes[i] * inner_size;
        const dst_base = o * block_size;
        for (let j = 0; j < block_size; j++) {
          grad_datas[i][dst_base + j] = dzData[dz_idx++];
        }
      }
    }

    for (let i = 0; i < tensors.length; i++) {
      this.next_functions[i].backward(
        new Tensor(grad_datas[i], {}, { shape: tensors[i].shape })
      );
    }
  }
}
registerOperation('cat', Cat);

// softmax

class Softmax extends TorchFunction {
  private dim: number;

  protected _forward(input: Tensor, dim: number): Tensor {
    this.dim = dim;
    const rg = resultRequiresGrad(input);
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);

    const shape = input.shape;
    const ndim = shape.length;
    const d = dim < 0 ? dim + ndim : dim;

    const inputData = input.data;
    const outputData = new Array(input.dataLength());

    const outer = shape.slice(0, d).reduce((a, b) => a * b, 1);
    const dimSize = shape[d];
    const inner = shape.slice(d + 1).reduce((a, b) => a * b, 1);

    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        let maxVal = -Infinity;
        for (let k = 0; k < dimSize; k++) {
          const idx = o * dimSize * inner + k * inner + i;
          if (inputData[idx] > maxVal) maxVal = inputData[idx];
        }
        let sumExp = 0;
        for (let k = 0; k < dimSize; k++) {
          const idx = o * dimSize * inner + k * inner + i;
          sumExp += Math.exp(inputData[idx] - maxVal);
        }
        for (let k = 0; k < dimSize; k++) {
          const idx = o * dimSize * inner + k * inner + i;
          outputData[idx] = Math.exp(inputData[idx] - maxVal) / sumExp;
        }
      }
    }

    const result = new Tensor(outputData, { requires_grad: rg }, { operation: rg ? this : null, shape: [...shape] });
    if (rg) {
      this.saved_tensors = [result]; // save output for efficient backward
    }
    return result;
  }

  protected _backward(dz: Tensor): void {
    const [softmaxOut] = this.saved_tensors;
    const [inputFn] = this.next_functions;

    const shape = softmaxOut.shape;
    const ndim = shape.length;
    const d = this.dim < 0 ? this.dim + ndim : this.dim;

    const sData = softmaxOut.data;
    const dzData = dz.data;

    const outer = shape.slice(0, d).reduce((a, b) => a * b, 1);
    const dimSize = shape[d];
    const inner = shape.slice(d + 1).reduce((a, b) => a * b, 1);

    const gradData = new Array(softmaxOut.dataLength());

    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        // dot(dz, softmax) along this slice
        let dotProd = 0;
        for (let k = 0; k < dimSize; k++) {
          const idx = o * dimSize * inner + k * inner + i;
          dotProd += dzData[idx] * sData[idx];
        }
        // grad = softmax * (dz - dot)
        for (let k = 0; k < dimSize; k++) {
          const idx = o * dimSize * inner + k * inner + i;
          gradData[idx] = sData[idx] * (dzData[idx] - dotProd);
        }
      }
    }

    inputFn.backward(new Tensor(gradData, {}, { shape: [...shape] }));
  }
}
registerOperation('softmax', Softmax);

// clamp

class Clamp extends TorchFunction {
  private min_val: number;
  private max_val: number;

  protected _forward(input: Tensor, min_val: number, max_val: number): Tensor {
    this.min_val = min_val;
    this.max_val = max_val;
    const rg = resultRequiresGrad(input);
    if (rg) {
      this.saved_tensors = [input];
    }
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);

    const inputData = input.data;
    const outputData = inputData.map(v => Math.min(Math.max(v, min_val), max_val));

    return new Tensor(outputData, { requires_grad: rg }, { operation: rg ? this : null, shape: [...input.shape] });
  }

  protected _backward(dz: Tensor): void {
    const [input] = this.saved_tensors;
    const [inputFn] = this.next_functions;
    const min_val = this.min_val;
    const max_val = this.max_val;

    const inputData = input.data;
    const dzData = dz.data;
    const gradData = inputData.map((v, i) =>
      (v > min_val && v < max_val) ? dzData[i] : 0
    );

    inputFn.backward(new Tensor(gradData, {}, { shape: [...input.shape] }));
  }
}
registerOperation('clamp', Clamp);
