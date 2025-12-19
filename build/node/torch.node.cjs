"use strict";
Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const gpu_js = require("@veehz/gpu.js");
function _broadcast_shape(a_shape, b_shape) {
  const result_length = Math.max(a_shape.length, b_shape.length);
  const padded_a_shape = [...Array(result_length - a_shape.length).fill(1), ...a_shape];
  const padded_b_shape = [...Array(result_length - b_shape.length).fill(1), ...b_shape];
  const result_shape = [];
  for (let i = 0; i < result_length; i++) {
    if (padded_a_shape[i] !== padded_b_shape[i] && padded_a_shape[i] !== 1 && padded_b_shape[i] !== 1) {
      throw new Error(`Shape mismatch: ${a_shape} and ${b_shape}`);
    }
    result_shape.push(Math.max(padded_a_shape[i], padded_b_shape[i]));
  }
  return result_shape;
}
function _pad_shape(shape, broadcast_shape) {
  if (shape.length >= broadcast_shape.length) {
    return shape;
  }
  return [...Array(broadcast_shape.length - shape.length).fill(1), ...shape];
}
function _get_original_index(original_shape, new_shape, index2) {
  let original_index = 0;
  let cur_stride = 1;
  let temp_index = index2;
  for (let i = original_shape.length - 1; i >= 0; i--) {
    if (original_shape[i] > 1) {
      const dim_index = temp_index % new_shape[i];
      original_index = original_index + dim_index * cur_stride;
    }
    cur_stride *= original_shape[i];
    temp_index = Math.floor(temp_index / new_shape[i]);
  }
  return original_index;
}
function _get_original_index_kernel(original_shape, new_shape, index2) {
  let original_index = 0;
  let cur_stride = 1;
  let temp_index = index2;
  for (let i = this.constants.shape_length - 1; i >= 0; i--) {
    if (original_shape[i] > 1) {
      const dim_index = temp_index % new_shape[i];
      original_index = original_index + dim_index * cur_stride;
    }
    cur_stride = cur_stride * original_shape[i];
    temp_index = Math.floor(temp_index / new_shape[i]);
  }
  return original_index;
}
function _get_original_index_from_transposed_index(original_shape, dim0, dim1, transposed_index) {
  let original_index = 0;
  let cur_stride = 1;
  let temp_index = transposed_index;
  let dim0_index = 0;
  let dim1_index = 0;
  for (let i = this.constants.shape_length - 1; i >= 0; i--) {
    const dim_index = temp_index % original_shape[i];
    if (i == dim0) {
      dim0_index = dim_index;
    }
    if (i == dim1) {
      dim1_index = dim_index;
    }
    temp_index = Math.floor(temp_index / original_shape[i]);
  }
  temp_index = transposed_index;
  for (let j = this.constants.shape_length - 1; j >= 0; j--) {
    const dim_index = temp_index % original_shape[j];
    if (j == dim0) {
      original_index = original_index + dim1_index * cur_stride;
    } else if (j == dim1) {
      original_index = original_index + dim0_index * cur_stride;
    } else {
      original_index = original_index + dim_index * cur_stride;
    }
    cur_stride = cur_stride * original_shape[j];
    temp_index = Math.floor(temp_index / original_shape[j]);
  }
  return original_index;
}
const gpu = new gpu_js.GPU({ mode: "cpu" });
gpu.addFunction(_get_original_index_kernel, {
  returnType: "Integer",
  argumentTypes: {
    original_shape: "Array",
    new_shape: "Array",
    index: "Integer"
  }
});
gpu.addFunction(_get_original_index_from_transposed_index, {
  returnType: "Integer",
  argumentTypes: {
    original_shape: "Array",
    dim0: "Integer",
    dim1: "Integer",
    transposed_index: "Integer"
  }
});
const operations = /* @__PURE__ */ new Map();
const operations_cache = /* @__PURE__ */ new Map();
function registerOperation(name, func) {
  operations.set(name, func);
}
function getOperation(name) {
  const func = operations.get(name);
  if (!func) {
    throw new Error(`Operation '${name}' is not registered.`);
  }
  return func;
}
function getOperationCache(name) {
  const operation = operations_cache.get(name);
  if (!operation) {
    operations_cache.set(name, new (getOperation(name))());
    return operations_cache.get(name);
  }
  return operation;
}
function _get_shape(data) {
  if (ArrayBuffer.isView(data)) {
    return [data.length];
  }
  const shape = [];
  while (Array.isArray(data)) {
    shape.push(data.length);
    data = data[0];
  }
  return shape;
}
function _flatten(data) {
  if (Array.isArray(data)) {
    return data.flatMap((item) => _flatten(item));
  } else if (ArrayBuffer.isView(data)) {
    return Array.from(data);
  } else {
    return [data];
  }
}
class Tensor {
  data;
  _shape;
  operation = null;
  grad = null;
  requires_grad;
  constructor(data, options = {}, internal_options = {}) {
    this.data = data instanceof gpu_js.Texture ? data : _flatten(data);
    this.requires_grad = options.requires_grad ?? false;
    this._shape = internal_options.shape ?? _get_shape(data instanceof gpu_js.Texture ? data.toArray() : data);
    this.operation = internal_options.operation ?? null;
  }
  // TODO: Somehow having a shape of [] will have a weird error:
  // TypeError: Cannot read properties of undefined (reading 'length')
  // when running kernel (something to do with constants?)
  // so a little hack to return [1] when the shape is []
  get shape() {
    return this._shape.length === 0 ? [1] : this._shape;
  }
  toArray_() {
    if (this.data instanceof gpu_js.Texture) {
      this.data = this.data.toArray();
    }
  }
  toArray() {
    if (this.data instanceof gpu_js.Texture) {
      return this.data.toArray();
    }
    return this.data;
  }
  dataLength() {
    if (this.data instanceof gpu_js.Texture) {
      return this.shape.reduce((acc, val) => acc * val, 1);
    }
    return this.data.length;
  }
  set shape(shape) {
    this._shape = shape;
  }
  _executeUnaryOp(opName) {
    const operation = this.requires_grad ? new (getOperation(opName))() : getOperationCache(opName);
    return operation.forward(this);
  }
  _executeBinaryOp(opName, other) {
    if (typeof other == "number") {
      other = new Tensor(other);
    }
    const operation = this.requires_grad || other.requires_grad ? new (getOperation(opName))() : getOperationCache(opName);
    return operation.forward(this, other);
  }
  _executeOpRaw(opName, ...args) {
    const operation = new (getOperation(opName))();
    return operation.forward(this, ...args);
  }
  item() {
    if (this.dataLength() !== 1) {
      throw new Error("Tensor.item() is only valid for scalars");
    }
    return this.toArray()[0];
  }
  detach() {
    return new Tensor(this.data, { requires_grad: false }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = false;
    this.grad = null;
    this.operation = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  backward(grad) {
    if (!this.requires_grad) {
      return;
    }
    if (!grad) {
      if (this.dataLength() !== 1) {
        throw new Error("Gradient is required for non-scalar tensors");
      }
      grad = new Tensor(1);
    } else {
      grad.toArray_();
    }
    if (!this.grad) {
      this.grad = new Tensor(Array(this.dataLength()).fill(0));
    }
    this.grad.toArray_();
    for (let i = 0; i < grad.dataLength(); i++) {
      this.grad.data[_get_original_index(this.shape, grad.shape, i)] += grad.data[i];
    }
    if (this.operation) {
      this.operation.backward(this.grad);
    }
  }
  // operations
  // binary pointwise
  add(other) {
    return this._executeBinaryOp("add", other);
  }
  sub(other) {
    return this._executeBinaryOp("sub", other);
  }
  mul(other) {
    return this._executeBinaryOp("mul", other);
  }
  div(other) {
    return this._executeBinaryOp("div", other);
  }
  pow(other) {
    if (typeof other == "number" && other % 1 === 0) {
      return this._executeOpRaw("powint", other);
    }
    return this._executeBinaryOp("pow", other);
  }
  fmod(other) {
    return this._executeBinaryOp("fmod", other);
  }
  // unary pointwise
  log() {
    return this._executeUnaryOp("log");
  }
  sqrt() {
    return this._executeUnaryOp("sqrt");
  }
  exp() {
    return this._executeUnaryOp("exp");
  }
  square() {
    return this._executeUnaryOp("square");
  }
  abs() {
    return this._executeUnaryOp("abs");
  }
  sign() {
    return this._executeUnaryOp("sign");
  }
  neg() {
    return this._executeUnaryOp("neg");
  }
  reciprocal() {
    return this._executeUnaryOp("reciprocal");
  }
  reshape(shape) {
    return this._executeOpRaw("reshape", shape);
  }
  unsqueeze(dim) {
    return this._executeOpRaw("unsqueeze", dim);
  }
  // trigonometric
  sin() {
    return this._executeUnaryOp("sin");
  }
  cos() {
    return this._executeUnaryOp("cos");
  }
  tan() {
    return this._executeUnaryOp("tan");
  }
  // reduction
  sum() {
    return this._executeUnaryOp("sum");
  }
  mean() {
    return this._executeUnaryOp("mean");
  }
  // linalg
  transpose(dim0, dim1) {
    return this._executeOpRaw("transpose", dim0, dim1);
  }
  matmul(other) {
    return this._executeBinaryOp("matmul", other);
  }
  // comparison
  lt(other) {
    return this._executeBinaryOp("lt", other);
  }
  gt(other) {
    return this._executeBinaryOp("gt", other);
  }
  le(other) {
    return this._executeBinaryOp("le", other);
  }
  ge(other) {
    return this._executeBinaryOp("ge", other);
  }
  eq(other) {
    return this._executeBinaryOp("eq", other);
  }
  ne(other) {
    return this._executeBinaryOp("ne", other);
  }
}
class Operation {
}
class UnaryOperation extends Operation {
}
class BinaryOperation extends Operation {
}
function generate_function(opname) {
  return (...args) => {
    const operation = new (getOperation(opname))();
    return operation.forward(...args);
  };
}
function generate_unary_function$1(opname) {
  return (a) => {
    if (typeof a == "number") {
      a = new Tensor(a);
    }
    const operation = new (getOperation(opname))();
    return operation.forward(a);
  };
}
function generate_binary_function(opname) {
  return (a, b) => {
    if (typeof a == "number") {
      a = new Tensor(a);
    }
    if (typeof b == "number") {
      b = new Tensor(b);
    }
    const operation = new (getOperation(opname))();
    return operation.forward(a, b);
  };
}
const add = generate_binary_function("add");
const sub = generate_binary_function("sub");
const mul = generate_binary_function("mul");
const div = generate_binary_function("div");
const pow = generate_binary_function("pow");
const fmod = generate_binary_function("fmod");
const log = generate_unary_function$1("log");
const sqrt = generate_unary_function$1("sqrt");
const exp = generate_unary_function$1("exp");
const square = generate_unary_function$1("square");
const abs = generate_unary_function$1("abs");
const sign = generate_unary_function$1("sign");
const neg = generate_unary_function$1("neg");
const reciprocal = generate_unary_function$1("reciprocal");
const reshape = generate_function("reshape");
const unsqueeze = generate_function("unsqueeze");
const sin = generate_unary_function$1("sin");
const cos = generate_unary_function$1("cos");
const tan = generate_unary_function$1("tan");
const sum = generate_unary_function$1("sum");
const mean = generate_unary_function$1("mean");
const transpose = generate_function("transpose");
const matmul = generate_binary_function("matmul");
const lt = generate_binary_function("lt");
const gt = generate_binary_function("gt");
const le = generate_binary_function("le");
const ge = generate_binary_function("ge");
const eq = generate_binary_function("eq");
const ne = generate_binary_function("ne");
const _add_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] + b[b_index];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _add_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _add_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Add extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _add_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
    a.backward(dz);
    b.backward(dz);
  }
}
registerOperation("add", Add);
const _sub_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] - b[b_index];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _sub_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _sub_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Sub extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _sub_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
    a.backward(dz);
    b.backward(dz.mul(new Tensor(-1)));
  }
}
registerOperation("sub", Sub);
const _mul_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] * b[b_index];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _mul_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _mul_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Mul extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _mul_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
    a.backward(dz.mul(b));
    b.backward(dz.mul(a));
  }
}
registerOperation("mul", Mul);
const _div_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] / b[b_index];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _div_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _div_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Div extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _div_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
    a.backward(dz.div(b));
    b.backward(dz.mul(a).mul(new Tensor(-1)).div(b).div(b));
  }
}
registerOperation("div", Div);
const _pow_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return Math.pow(a[a_index], b[b_index]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _pow_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _pow_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Pow extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _pow_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
    a.backward(dz.mul(b).mul(a.pow(b.sub(new Tensor(1)))));
    b.backward(dz.mul(a.pow(b)).mul(a.log()));
  }
}
registerOperation("pow", Pow);
const _fmod_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] % b[b_index];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _fmod_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _fmod_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Fmod extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _fmod_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
    a.backward(dz);
  }
}
registerOperation("fmod", Fmod);
function _powint_kernel_function(a, n) {
  return Math.pow(a[this.thread.x], n);
}
const _powint_kernel = gpu.createKernel(
  _powint_kernel_function,
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _powint_tensor(a, n, operation = null) {
  const kernel = _powint_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, n),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class PowInt extends Operation {
  cache;
  forward(a, n) {
    if (a.requires_grad) {
      this.cache = [a, n];
    }
    return _powint_tensor(a, n, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, n] = this.cache;
    a.backward(dz.mul(n).mul(a.pow(n - 1)));
  }
}
registerOperation("powint", PowInt);
const _log_kernel = gpu.createKernel(
  function(a) {
    return Math.log(a[this.thread.x]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _log_tensor(a, operation = null) {
  const kernel = _log_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Log extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _log_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(new Tensor(1).div(a));
  }
}
registerOperation("log", Log);
const _sqrt_kernel = gpu.createKernel(
  function(a) {
    return Math.sqrt(a[this.thread.x]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _sqrt_tensor(a, operation = null) {
  const kernel = _sqrt_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Sqrt extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _sqrt_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(new Tensor(1).div(a.sqrt()).div(2));
  }
}
registerOperation("sqrt", Sqrt);
const _exp_kernel = gpu.createKernel(
  function(a) {
    return Math.exp(a[this.thread.x]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _exp_tensor(a, operation = null) {
  const kernel = _exp_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Exp extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _exp_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(a.exp()));
  }
}
registerOperation("exp", Exp);
const _square_kernel = gpu.createKernel(
  function(a) {
    return a[this.thread.x] * a[this.thread.x];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _square_tensor(a, operation = null) {
  const kernel = _square_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Square extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _square_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(a).mul(new Tensor(2)));
  }
}
registerOperation("square", Square);
const _abs_kernel = gpu.createKernel(
  function(a) {
    return Math.abs(a[this.thread.x]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _abs_tensor(a, operation = null) {
  const kernel = _abs_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Abs extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _abs_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(sign(a)));
  }
}
registerOperation("abs", Abs);
const _sign_kernel = gpu.createKernel(
  function(a) {
    return Math.sign(a[this.thread.x]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _sign_tensor(a, operation = null) {
  const kernel = _sign_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Sign extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _sign_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
  }
}
registerOperation("sign", Sign);
const _neg_kernel = gpu.createKernel(
  function(a) {
    return Math.sign(a[this.thread.x]) * a[this.thread.x];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _neg_tensor(a, operation = null) {
  const kernel = _neg_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Neg extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _neg_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(new Tensor(-1)));
  }
}
registerOperation("neg", Neg);
const _reciprocal_kernel = gpu.createKernel(
  function(a) {
    return 1 / a[this.thread.x];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _reciprocal_tensor(a, operation = null) {
  const kernel = _reciprocal_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Reciprocal extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _reciprocal_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(a.pow(-2)));
  }
}
registerOperation("reciprocal", Reciprocal);
class Reshape extends Operation {
  cache;
  forward(a, shape) {
    const previous_length = a.dataLength();
    const target_length = shape.reduce((acc, val) => acc * val, 1);
    if (previous_length !== target_length) {
      throw new Error("Shape mismatch: " + a.shape + " and " + shape);
    }
    if (a.requires_grad) {
      this.cache = [a];
    }
    return new Tensor(
      a.data,
      { requires_grad: a.requires_grad },
      { operation: a.requires_grad ? this : null, shape }
    );
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.reshape(a.shape));
  }
}
registerOperation("reshape", Reshape);
class Unsqueeze extends Operation {
  cache;
  forward(a, dim) {
    if (a.requires_grad) {
      this.cache = [a];
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
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.reshape(a.shape));
  }
}
registerOperation("unsqueeze", Unsqueeze);
const _sin_kernel = gpu.createKernel(
  function(a) {
    return Math.sin(a[this.thread.x]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _sin_tensor(a, operation = null) {
  const kernel = _sin_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Sin extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _sin_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(a.cos()));
  }
}
registerOperation("sin", Sin);
const _cos_kernel = gpu.createKernel(
  function(a) {
    return Math.cos(a[this.thread.x]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _cos_tensor(a, operation = null) {
  const kernel = _cos_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Cos extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _cos_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(a.sin().neg()));
  }
}
registerOperation("cos", Cos);
const _tan_kernel = gpu.createKernel(
  function(a) {
    return Math.tan(a[this.thread.x]);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _tan_tensor(a, operation = null) {
  const kernel = _tan_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Tan extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _tan_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(a.cos().pow(-2)));
  }
}
registerOperation("tan", Tan);
function _sum_tensor(a, operation = null) {
  return new Tensor(
    a.toArray().reduce((acc, val) => acc + val, 0),
    { requires_grad: a.requires_grad },
    { operation }
  );
}
class Sum extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _sum_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    const result = new Tensor(Array(a.dataLength()).fill(dz.item()));
    a.backward(result);
  }
}
registerOperation("sum", Sum);
function _mean_tensor(a, operation = null) {
  return new Tensor(
    a.toArray().reduce((acc, val) => acc + val, 0) / a.dataLength(),
    { requires_grad: a.requires_grad },
    { operation }
  );
}
class Mean extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _mean_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    const result = new Tensor(Array(a.dataLength()).fill(dz.item() / a.dataLength()));
    a.backward(result);
  }
}
registerOperation("mean", Mean);
const _transpose_kernel = gpu.createKernel(
  function(a, as, dim0, dim1) {
    const a_index = _get_original_index_from_transposed_index(as, dim0, dim1, this.thread.x);
    return a[a_index];
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _transpose_tensor(a, dim0, dim1, operation = null) {
  const kernel = _transpose_kernel;
  kernel.setConstants({
    shape_length: a.shape.length
  });
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  const swapped_shape = [...a.shape];
  [swapped_shape[dim0], swapped_shape[dim1]] = [swapped_shape[dim1], swapped_shape[dim0]];
  return new Tensor(
    kernel(a.data, a.shape, dim0, dim1),
    { requires_grad: a.requires_grad },
    { operation, shape: swapped_shape }
  );
}
class Transpose extends Operation {
  cache;
  forward(a, dim0, dim1) {
    this.cache = [a, dim0, dim1];
    return _transpose_tensor(a, dim0, dim1, this);
  }
  backward(dz) {
    const [a, dim0, dim1] = this.cache;
    a.backward(dz.transpose(dim0, dim1));
  }
}
registerOperation("transpose", Transpose);
function _matmul_kernel_function(a, as, b, bs, bcs) {
  let a_index = _get_original_index_kernel(as, bcs, this.thread.x);
  let b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
  const l = this.constants.shape_length;
  const tmp1 = bcs[l] * bcs[l + 1];
  const position = this.thread.x % tmp1;
  a_index = a_index * as[l] * as[l + 1] + Math.floor(position / bcs[l + 1]) * as[l + 1];
  b_index = b_index * bs[l] * bs[l + 1] + position % bcs[l + 1];
  const b_stride = bs[l + 1];
  let sum2 = 0;
  for (let i = 0; i < this.constants.lp; i++) {
    sum2 = sum2 + a[a_index] * b[b_index];
    a_index = a_index + 1;
    b_index = b_index + b_stride;
  }
  return sum2;
}
const _matmul_kernel = gpu.createKernel(
  _matmul_kernel_function,
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _matmul_tensor(a, b, operation = null) {
  if (a.shape.length == 1 && b.shape.length == 1) {
    return a.mul(b).sum();
  }
  const a_1d = a.shape.length == 1;
  const b_1d = b.shape.length == 1;
  const a_shape = a_1d ? [1, a.shape[0]] : a.shape;
  const b_shape = b_1d ? [b.shape[0], 1] : b.shape;
  if (a_shape[a_shape.length - 1] != b_shape[b_shape.length - 2]) {
    throw new Error("Shape mismatch: " + a.shape + " and " + b.shape);
  }
  const loop_iterations = a_shape[a_shape.length - 1];
  if (loop_iterations > 1e3) {
    throw new Error("Loop iterations too large: " + loop_iterations);
  }
  const broadcast_shape = _broadcast_shape(a_shape.slice(0, -2), b_shape.slice(0, -2)).concat([
    a_shape[a_shape.length - 2],
    b_shape[b_shape.length - 1]
  ]);
  const padded_a_shape = _pad_shape(a_shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b_shape, broadcast_shape);
  const kernel = _matmul_kernel;
  kernel.setConstants({
    lp: loop_iterations,
    // assumes that _get_original_index_kernel reads from the front
    shape_length: broadcast_shape.length - 2
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  let shape_after_removing_extra_dims = [...broadcast_shape];
  if (a_1d) {
    shape_after_removing_extra_dims = shape_after_removing_extra_dims.slice(0, -2).concat([broadcast_shape[broadcast_shape.length - 1]]);
  }
  if (b_1d) {
    shape_after_removing_extra_dims = shape_after_removing_extra_dims.slice(0, -1);
  }
  return new Tensor(
    kernel(
      a.data,
      padded_a_shape,
      b.data,
      padded_b_shape,
      broadcast_shape
    ),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: shape_after_removing_extra_dims }
  );
}
class Matmul extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _matmul_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
  }
}
registerOperation("matmul", Matmul);
const _lt_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] < b[b_index] ? 1 : 0;
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _lt_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _lt_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Lt extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _lt_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
  }
}
registerOperation("lt", Lt);
const _gt_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] > b[b_index] ? 1 : 0;
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _gt_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _gt_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Gt extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _gt_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
  }
}
registerOperation("gt", Gt);
const _le_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] <= b[b_index] ? 1 : 0;
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _le_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _le_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Le extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _le_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
  }
}
registerOperation("le", Le);
const _ge_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] >= b[b_index] ? 1 : 0;
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _ge_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _ge_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Ge extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _ge_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
  }
}
registerOperation("ge", Ge);
const _eq_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] == b[b_index] ? 1 : 0;
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _eq_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _eq_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Eq extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _eq_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
  }
}
registerOperation("eq", Eq);
const _ne_kernel = gpu.createKernel(
  function(a, as, b, bs, bcs) {
    const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
    const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);
    return a[a_index] != b[b_index] ? 1 : 0;
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _ne_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _ne_kernel;
  kernel.setConstants({
    shape_length: broadcast_shape.length
  });
  kernel.setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
class Ne extends BinaryOperation {
  cache;
  forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.cache = [a, b];
    }
    return _ne_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  backward(dz) {
    const [a, b] = this.cache;
  }
}
registerOperation("ne", Ne);
function get_shape_from_args(args) {
  if (Array.isArray(args[0])) {
    return args[0];
  }
  return args;
}
function randn(...args) {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(Math.random()));
  tensor.shape = shape;
  return tensor;
}
function rand(...args) {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(Math.random()));
  tensor.shape = shape;
  return tensor;
}
function randint(low, high, shape) {
  const tensor = new Tensor(
    Array(shape.reduce((a, b) => a * b, 1)).fill(Math.floor(Math.random() * (high - low) + low))
  );
  tensor.shape = shape;
  return tensor;
}
function ones(...args) {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(1));
  tensor.shape = shape;
  return tensor;
}
function zeros(...args) {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(0));
  tensor.shape = shape;
  return tensor;
}
function linspace(start, end, steps) {
  const data = [];
  const step = (end - start) / (steps - 1);
  for (let i = 0; i < steps - 1; i++) {
    data.push(start + i * step);
  }
  data.push(end);
  return new Tensor(data);
}
function arange(start, end = void 0, step = 1) {
  const data = [];
  for (let i = start; i < end; i += step) {
    data.push(i);
  }
  return new Tensor(data);
}
const _relu_kernel = gpu.createKernel(
  function(a) {
    return Math.max(a[this.thread.x], 0);
  },
  {
    dynamicOutput: true,
    dynamicArguments: true
    // pipeline: true,
    // immutable: true
  }
);
function _relu_tensor(a, operation = null) {
  const kernel = _relu_kernel;
  kernel.setOutput([a.shape.reduce((acc, val) => acc * val, 1)]);
  return new Tensor(
    kernel(a.data),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Relu extends UnaryOperation {
  cache;
  forward(a) {
    if (a.requires_grad) {
      this.cache = [a];
    }
    return _relu_tensor(a, a.requires_grad ? this : null);
  }
  backward(dz) {
    const [a] = this.cache;
    a.backward(dz.mul(a.gt(0)));
  }
}
registerOperation("relu", Relu);
class Parameter extends Tensor {
  constructor(data, options = {
    requires_grad: true
  }, internal_options = {}) {
    if (data instanceof Tensor) {
      super(data.data, { requires_grad: true }, { shape: data.shape });
    } else if (data instanceof Parameter) {
      super(data.data, { requires_grad: true }, { shape: data.shape });
    } else {
      super(data, options, internal_options);
    }
  }
}
class Module {
  _modules;
  _parameters;
  constructor() {
    this._parameters = {};
    this._modules = {};
  }
  register_parameter(parameter_name, parameter) {
    this._parameters[parameter_name] = parameter;
  }
  register_module(module_name, module2) {
    this._modules[module_name] = module2;
  }
  register(name, value) {
    if (value instanceof Parameter) {
      this.register_parameter(name, value);
    } else {
      this.register_module(name, value);
    }
  }
  parameters() {
    let params = Object.values(this._parameters);
    for (const module2 of Object.values(this._modules)) {
      params = params.concat(module2.parameters());
    }
    return params;
  }
}
class Linear extends Module {
  weight;
  bias;
  constructor(in_features, out_features) {
    super();
    const k = Math.sqrt(1 / in_features);
    this.weight = new Parameter(rand([out_features, in_features]).mul(2 * k).sub(k));
    this.bias = new Parameter(rand([out_features]).mul(2 * k).sub(k));
    this.register("weight", this.weight);
    this.register("bias", this.bias);
  }
  forward(input) {
    return input.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
}
class ReLU extends Module {
  constructor() {
    super();
  }
  forward(input) {
    return relu(input);
  }
}
function generate_unary_function(opname) {
  return (a) => {
    if (typeof a == "number") {
      a = new Tensor(a);
    }
    const operation = new (getOperation(opname))();
    return operation.forward(a);
  };
}
const relu = generate_unary_function("relu");
const functional = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu
}, Symbol.toStringTag, { value: "Module" }));
const index$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Linear,
  Module,
  Parameter,
  ReLU,
  functional
}, Symbol.toStringTag, { value: "Module" }));
class Optimizer {
  params;
  defaults;
  constructor(params, defaults) {
    this.params = params;
    this.defaults = defaults;
  }
  zero_grad() {
    for (const param of this.params) {
      param.grad = null;
    }
  }
}
class SGD extends Optimizer {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(params, lr = 1e-3, momentum = 0, dampening = 0, weight_decay = 0, nesterov = false, maximize = false) {
    super(params, {});
    this.lr = lr;
    this.momentum = momentum;
    this.dampening = dampening;
    this.weight_decay = weight_decay;
    this.nesterov = nesterov;
    this.maximize = maximize;
  }
  step() {
    for (const param of this.params) {
      let g = this.maximize ? param.grad.mul(-1) : param.grad;
      if (this.weight_decay != 0) {
        g = g.add(param.mul(this.weight_decay));
      }
      if (this.momentum != 0) {
        let dampening = this.dampening;
        if (!this.state.has(param)) {
          this.state.set(param, { velocity: zeros(param.shape) });
          dampening = 0;
        }
        let buf = this.state.get(param).velocity;
        buf = buf.mul(this.momentum);
        buf = buf.add(g.mul(dampening));
        if (this.nesterov) {
          g = g.add(buf.mul(this.momentum));
        } else {
          g = g.add(buf);
        }
        this.state.set(param, { velocity: buf });
      }
      const newParam = param.sub(g.mul(this.lr));
      param.data = newParam.data;
    }
  }
}
const index = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Optimizer,
  SGD
}, Symbol.toStringTag, { value: "Module" }));
Object.defineProperty(exports, "GPU", {
  enumerable: true,
  get: () => gpu_js.GPU
});
exports.Abs = Abs;
exports.Add = Add;
exports.Cos = Cos;
exports.Div = Div;
exports.Eq = Eq;
exports.Exp = Exp;
exports.Fmod = Fmod;
exports.Ge = Ge;
exports.Gt = Gt;
exports.Le = Le;
exports.Log = Log;
exports.Lt = Lt;
exports.Matmul = Matmul;
exports.Mean = Mean;
exports.Mul = Mul;
exports.Ne = Ne;
exports.Neg = Neg;
exports.Pow = Pow;
exports.PowInt = PowInt;
exports.Reciprocal = Reciprocal;
exports.Reshape = Reshape;
exports.Sign = Sign;
exports.Sin = Sin;
exports.Sqrt = Sqrt;
exports.Square = Square;
exports.Sub = Sub;
exports.Sum = Sum;
exports.Tan = Tan;
exports.Tensor = Tensor;
exports.Transpose = Transpose;
exports.Unsqueeze = Unsqueeze;
exports.abs = abs;
exports.add = add;
exports.arange = arange;
exports.cos = cos;
exports.div = div;
exports.eq = eq;
exports.exp = exp;
exports.fmod = fmod;
exports.ge = ge;
exports.gpu = gpu;
exports.gt = gt;
exports.le = le;
exports.linspace = linspace;
exports.log = log;
exports.lt = lt;
exports.matmul = matmul;
exports.mean = mean;
exports.mul = mul;
exports.ne = ne;
exports.neg = neg;
exports.nn = index$1;
exports.ones = ones;
exports.optim = index;
exports.pow = pow;
exports.rand = rand;
exports.randint = randint;
exports.randn = randn;
exports.reciprocal = reciprocal;
exports.reshape = reshape;
exports.sign = sign;
exports.sin = sin;
exports.sqrt = sqrt;
exports.square = square;
exports.sub = sub;
exports.sum = sum;
exports.tan = tan;
exports.transpose = transpose;
exports.unsqueeze = unsqueeze;
exports.zeros = zeros;
