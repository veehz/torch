var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var _a;
let globalId = 0;
const getNextId = /* @__PURE__ */ __name(() => {
  return globalId++;
}, "getNextId");
const eventBus = new EventTarget();
const _Operation = class _Operation {
  id = getNextId();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...args) {
    eventBus.dispatchEvent(new CustomEvent("operation.beforeForward", {
      detail: {
        operation: this,
        args
      }
    }));
    const result = this._forward(...args);
    eventBus.dispatchEvent(new CustomEvent("operation.afterForward", {
      detail: {
        operation: this,
        args,
        result,
        requires_grad: result.requires_grad
      }
    }));
    return result;
  }
  backward(dz) {
    eventBus.dispatchEvent(new CustomEvent("operation.beforeBackward", { detail: { operation: this, dz } }));
    for (const x of this._retained_tensors) {
      if (!x.grad) {
        x.grad = new Tensor(new Array(x.dataLength()).fill(0));
      }
      x.grad = x.grad.add(dz);
    }
    this._backward(dz);
    eventBus.dispatchEvent(new CustomEvent("operation.afterBackward", { detail: { operation: this, dz } }));
  }
};
__name(_Operation, "Operation");
let Operation = _Operation;
const _NullOp = class _NullOp extends Operation {
  _forward(...args) {
    throw new Error("NullOp should not be called");
  }
  _backward(dz) {
    return;
  }
};
__name(_NullOp, "NullOp");
let NullOp = _NullOp;
const nullOp = new NullOp();
const _UnaryOperation = class _UnaryOperation extends Operation {
};
__name(_UnaryOperation, "UnaryOperation");
let UnaryOperation = _UnaryOperation;
const _BinaryOperation = class _BinaryOperation extends Operation {
};
__name(_BinaryOperation, "BinaryOperation");
let BinaryOperation = _BinaryOperation;
const _AccumulateGrad = class _AccumulateGrad extends UnaryOperation {
  variable;
  _forward(variable) {
    this.variable = variable;
    return variable;
  }
  _backward(dz) {
    if (!this.variable.grad) {
      this.variable.grad = new Tensor(new Array(this.variable.dataLength()).fill(0));
    }
    eventBus.dispatchEvent(new CustomEvent("operation.accumulateGrad", { detail: { operation: this, dz } }));
    this.variable.grad = this.variable.grad.add(dz);
  }
};
__name(_AccumulateGrad, "AccumulateGrad");
let AccumulateGrad = _AccumulateGrad;
const operations = /* @__PURE__ */ new Map();
const operations_cache = /* @__PURE__ */ new Map();
function registerOperation(name, func) {
  operations.set(name, func);
}
__name(registerOperation, "registerOperation");
function getOperation(name) {
  const func = operations.get(name);
  if (!func) {
    throw new Error(`Operation '${name}' is not registered.`);
  }
  return func;
}
__name(getOperation, "getOperation");
function getOperationCache(name) {
  const operation = operations_cache.get(name);
  if (!operation) {
    operations_cache.set(name, new (getOperation(name))());
    return operations_cache.get(name);
  }
  return operation;
}
__name(getOperationCache, "getOperationCache");
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
__name(_get_shape, "_get_shape");
function _flatten(data) {
  if (Array.isArray(data)) {
    return data.flatMap((item) => _flatten(item));
  } else if (ArrayBuffer.isView(data)) {
    return Array.from(data);
  } else {
    return [data];
  }
}
__name(_flatten, "_flatten");
const _Tensor = class _Tensor {
  id = getNextId();
  data;
  _shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(data, options = {}, internal_options = {}) {
    this.data = _flatten(data);
    this.requires_grad = options.requires_grad ?? false;
    this._shape = internal_options.shape ?? _get_shape(data);
    this.grad_fn = internal_options.operation ?? null;
    if (this.requires_grad && !this.grad_fn) {
      const acc = new AccumulateGrad();
      acc.variable = this;
      this.grad_fn = acc;
    }
  }
  // TODO: Somehow having a shape of [] will have a weird error:
  // TypeError: Cannot read properties of undefined (reading 'length')
  // when running kernel (something to do with constants?)
  // so a little hack to return [1] when the shape is []
  get shape() {
    return this._shape.length === 0 ? [1] : this._shape;
  }
  toArray_() {
    return;
  }
  toArray() {
    return this.data;
  }
  dataLength() {
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
      other = new _Tensor(other);
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
    return new _Tensor(this.data, { requires_grad: false }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = false;
    this.grad = null;
    this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = false;
  retain_grad() {
    if (this.grad_fn instanceof AccumulateGrad) return;
    if (this.is_retain_grad) return;
    this.is_retain_grad = true;
    this.grad_fn._retained_tensors.push(this);
  }
  backward(grad) {
    if (!this.requires_grad) {
      return;
    }
    if (!grad) {
      if (this.dataLength() !== 1) {
        throw new Error("Gradient is required for non-scalar tensors");
      }
      grad = new _Tensor(1);
    } else {
      grad.toArray_();
    }
    if (this.grad_fn) {
      eventBus.dispatchEvent(new CustomEvent("tensor.beforeBackward", { detail: { tensor: this } }));
      this.grad_fn.backward(grad);
      eventBus.dispatchEvent(new CustomEvent("tensor.afterBackward", { detail: { tensor: this } }));
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
  maximum(other) {
    return this._executeBinaryOp("maximum", other);
  }
  minimum(other) {
    return this._executeBinaryOp("minimum", other);
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
};
__name(_Tensor, "Tensor");
let Tensor = _Tensor;
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
__name(_broadcast_shape, "_broadcast_shape");
function _pad_shape(shape, broadcast_shape) {
  if (shape.length >= broadcast_shape.length) {
    return shape;
  }
  return [...Array(broadcast_shape.length - shape.length).fill(1), ...shape];
}
__name(_pad_shape, "_pad_shape");
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
__name(_get_original_index, "_get_original_index");
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
__name(_get_original_index_kernel, "_get_original_index_kernel");
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
__name(_get_original_index_from_transposed_index, "_get_original_index_from_transposed_index");
function generate_function$1(opname) {
  return (...args) => {
    const operation = new (getOperation(opname))();
    return operation.forward(...args);
  };
}
__name(generate_function$1, "generate_function$1");
function generate_unary_function$1(opname) {
  return (a) => {
    if (typeof a == "number") {
      a = new Tensor(a);
    }
    const operation = new (getOperation(opname))();
    return operation.forward(a);
  };
}
__name(generate_unary_function$1, "generate_unary_function$1");
function generate_binary_function$1(opname) {
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
__name(generate_binary_function$1, "generate_binary_function$1");
const __left_index__ = generate_binary_function$1("__left_index__");
const __right_index__ = generate_binary_function$1("__right_index__");
const add = generate_binary_function$1("add");
const sub = generate_binary_function$1("sub");
const mul = generate_binary_function$1("mul");
const div = generate_binary_function$1("div");
const pow = generate_binary_function$1("pow");
const fmod = generate_binary_function$1("fmod");
const maximum = generate_binary_function$1("maximum");
const minimum = generate_binary_function$1("minimum");
const log = generate_unary_function$1("log");
const sqrt = generate_unary_function$1("sqrt");
const exp = generate_unary_function$1("exp");
const square = generate_unary_function$1("square");
const abs = generate_unary_function$1("abs");
const sign = generate_unary_function$1("sign");
const neg = generate_unary_function$1("neg");
const reciprocal = generate_unary_function$1("reciprocal");
const reshape = generate_function$1("reshape");
const unsqueeze = generate_function$1("unsqueeze");
const sin = generate_unary_function$1("sin");
const cos = generate_unary_function$1("cos");
const tan = generate_unary_function$1("tan");
const sum = generate_unary_function$1("sum");
const mean = generate_unary_function$1("mean");
const transpose = generate_function$1("transpose");
const matmul = generate_binary_function$1("matmul");
const lt = generate_binary_function$1("lt");
const gt = generate_binary_function$1("gt");
const le = generate_binary_function$1("le");
const ge = generate_binary_function$1("ge");
const eq = generate_binary_function$1("eq");
const ne = generate_binary_function$1("ne");
const ___left_index___kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a_index;
  }
  return res;
}, "___left_index___kernel");
function ___left_index___tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = ___left_index___kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(___left_index___tensor, "___left_index___tensor");
const ___Left_index__ = class ___Left_index__ extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return ___left_index___tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(___Left_index__, "__Left_index__");
let __Left_index__ = ___Left_index__;
registerOperation("__left_index__", __Left_index__);
const ___right_index___kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = b_index;
  }
  return res;
}, "___right_index___kernel");
function ___right_index___tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = ___right_index___kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(___right_index___tensor, "___right_index___tensor");
const ___Right_index__ = class ___Right_index__ extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return ___right_index___tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(___Right_index__, "__Right_index__");
let __Right_index__ = ___Right_index__;
registerOperation("__right_index__", __Right_index__);
const _add_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] + b[b_index];
  }
  return res;
}, "_add_kernel");
function _add_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _add_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_add_tensor, "_add_tensor");
const _Add = class _Add extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _add_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    aFn.backward(dz);
    bFn.backward(dz);
  }
};
__name(_Add, "Add");
let Add = _Add;
registerOperation("add", Add);
const _sub_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] - b[b_index];
  }
  return res;
}, "_sub_kernel");
function _sub_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _sub_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_sub_tensor, "_sub_tensor");
const _Sub = class _Sub extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _sub_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    aFn.backward(dz);
    bFn.backward(dz.mul(new Tensor(-1)));
  }
};
__name(_Sub, "Sub");
let Sub = _Sub;
registerOperation("sub", Sub);
const _mul_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] * b[b_index];
  }
  return res;
}, "_mul_kernel");
function _mul_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _mul_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_mul_tensor, "_mul_tensor");
const _Mul = class _Mul extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _mul_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    aFn.backward(dz.mul(b));
    bFn.backward(dz.mul(a));
  }
};
__name(_Mul, "Mul");
let Mul = _Mul;
registerOperation("mul", Mul);
const _div_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] / b[b_index];
  }
  return res;
}, "_div_kernel");
function _div_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _div_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_div_tensor, "_div_tensor");
const _Div = class _Div extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _div_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    aFn.backward(dz.div(b));
    bFn.backward(dz.mul(a).mul(new Tensor(-1)).div(b).div(b));
  }
};
__name(_Div, "Div");
let Div = _Div;
registerOperation("div", Div);
const _pow_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = Math.pow(a[a_index], b[b_index]);
  }
  return res;
}, "_pow_kernel");
function _pow_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _pow_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_pow_tensor, "_pow_tensor");
const _Pow = class _Pow extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _pow_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    aFn.backward(dz.mul(b).mul(a.pow(b.sub(new Tensor(1)))));
    bFn.backward(dz.mul(a.pow(b)).mul(a.log()));
  }
};
__name(_Pow, "Pow");
let Pow = _Pow;
registerOperation("pow", Pow);
const _fmod_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] % b[b_index];
  }
  return res;
}, "_fmod_kernel");
function _fmod_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _fmod_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_fmod_tensor, "_fmod_tensor");
const _Fmod = class _Fmod extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _fmod_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    aFn.backward(dz);
  }
};
__name(_Fmod, "Fmod");
let Fmod = _Fmod;
registerOperation("fmod", Fmod);
const _maximum_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = Math.max(a[a_index], b[b_index]);
  }
  return res;
}, "_maximum_kernel");
function _maximum_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _maximum_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_maximum_tensor, "_maximum_tensor");
const _Maximum = class _Maximum extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _maximum_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    aFn.backward(dz.mul(a.ge(b)));
    bFn.backward(dz.mul(b.gt(a)));
  }
};
__name(_Maximum, "Maximum");
let Maximum = _Maximum;
registerOperation("maximum", Maximum);
const _minimum_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = Math.min(a[a_index], b[b_index]);
  }
  return res;
}, "_minimum_kernel");
function _minimum_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _minimum_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_minimum_tensor, "_minimum_tensor");
const _Minimum = class _Minimum extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _minimum_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    aFn.backward(dz.mul(a.le(b)));
    bFn.backward(dz.mul(b.lt(a)));
  }
};
__name(_Minimum, "Minimum");
let Minimum = _Minimum;
registerOperation("minimum", Minimum);
function _powint_tensor(a, n, operation = null) {
  const data = new Array(a.dataLength());
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.pow(a.data[i], n);
  }
  return new Tensor(
    data,
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_powint_tensor, "_powint_tensor");
const _PowInt = class _PowInt extends Operation {
  n;
  _forward(a, n) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
      this.n = n;
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _powint_tensor(a, n, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const n = this.n;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(n).mul(a.pow(n - 1)));
  }
};
__name(_PowInt, "PowInt");
let PowInt = _PowInt;
registerOperation("powint", PowInt);
const _log_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.log(a[x]);
  }
  return res;
}, "_log_kernel");
function _log_tensor(a, operation = null) {
  const kernel = _log_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_log_tensor, "_log_tensor");
const _Log = class _Log extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _log_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(new Tensor(1).div(a));
  }
};
__name(_Log, "Log");
let Log = _Log;
registerOperation("log", Log);
const _sqrt_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.sqrt(a[x]);
  }
  return res;
}, "_sqrt_kernel");
function _sqrt_tensor(a, operation = null) {
  const kernel = _sqrt_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_sqrt_tensor, "_sqrt_tensor");
const _Sqrt = class _Sqrt extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _sqrt_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(new Tensor(1).div(a.sqrt()).div(2));
  }
};
__name(_Sqrt, "Sqrt");
let Sqrt = _Sqrt;
registerOperation("sqrt", Sqrt);
const _exp_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.exp(a[x]);
  }
  return res;
}, "_exp_kernel");
function _exp_tensor(a, operation = null) {
  const kernel = _exp_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_exp_tensor, "_exp_tensor");
const _Exp = class _Exp extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _exp_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(a.exp()));
  }
};
__name(_Exp, "Exp");
let Exp = _Exp;
registerOperation("exp", Exp);
const _square_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = a[x] * a[x];
  }
  return res;
}, "_square_kernel");
function _square_tensor(a, operation = null) {
  const kernel = _square_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_square_tensor, "_square_tensor");
const _Square = class _Square extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _square_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(a).mul(new Tensor(2)));
  }
};
__name(_Square, "Square");
let Square = _Square;
registerOperation("square", Square);
const _abs_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.abs(a[x]);
  }
  return res;
}, "_abs_kernel");
function _abs_tensor(a, operation = null) {
  const kernel = _abs_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_abs_tensor, "_abs_tensor");
const _Abs = class _Abs extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _abs_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(sign(a)));
  }
};
__name(_Abs, "Abs");
let Abs = _Abs;
registerOperation("abs", Abs);
const _sign_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.sign(a[x]);
  }
  return res;
}, "_sign_kernel");
function _sign_tensor(a, operation = null) {
  const kernel = _sign_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_sign_tensor, "_sign_tensor");
const _Sign = class _Sign extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _sign_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
  }
};
__name(_Sign, "Sign");
let Sign = _Sign;
registerOperation("sign", Sign);
const _neg_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = -a[x];
  }
  return res;
}, "_neg_kernel");
function _neg_tensor(a, operation = null) {
  const kernel = _neg_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_neg_tensor, "_neg_tensor");
const _Neg = class _Neg extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _neg_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(new Tensor(-1)));
  }
};
__name(_Neg, "Neg");
let Neg = _Neg;
registerOperation("neg", Neg);
const _reciprocal_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = 1 / a[x];
  }
  return res;
}, "_reciprocal_kernel");
function _reciprocal_tensor(a, operation = null) {
  const kernel = _reciprocal_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_reciprocal_tensor, "_reciprocal_tensor");
const _Reciprocal = class _Reciprocal extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _reciprocal_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(a.pow(-2)));
  }
};
__name(_Reciprocal, "Reciprocal");
let Reciprocal = _Reciprocal;
registerOperation("reciprocal", Reciprocal);
const _Reshape = class _Reshape extends Operation {
  _forward(a, shape) {
    const previous_length = a.dataLength();
    const target_length = shape.reduce((acc, val) => acc * val, 1);
    if (previous_length !== target_length) {
      throw new Error("Shape mismatch: " + a.shape + " and " + shape);
    }
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    if (a.grad_fn) {
      this.next_functions.push(a.grad_fn);
    } else if (a.requires_grad) {
      const acc = new AccumulateGrad();
      acc.variable = a;
      this.next_functions.push(acc);
    } else {
      this.next_functions.push(nullOp);
    }
    return new Tensor(
      a.data,
      { requires_grad: a.requires_grad },
      { operation: a.requires_grad ? this : null, shape }
    );
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.reshape(a.shape));
  }
};
__name(_Reshape, "Reshape");
let Reshape = _Reshape;
registerOperation("reshape", Reshape);
const _Unsqueeze = class _Unsqueeze extends Operation {
  _forward(a, dim) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    if (a.grad_fn) {
      this.next_functions.push(a.grad_fn);
    } else if (a.requires_grad) {
      const acc = new AccumulateGrad();
      acc.variable = a;
      this.next_functions.push(acc);
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
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.reshape(a.shape));
  }
};
__name(_Unsqueeze, "Unsqueeze");
let Unsqueeze = _Unsqueeze;
registerOperation("unsqueeze", Unsqueeze);
const _sin_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.sin(a[x]);
  }
  return res;
}, "_sin_kernel");
function _sin_tensor(a, operation = null) {
  const kernel = _sin_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_sin_tensor, "_sin_tensor");
const _Sin = class _Sin extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _sin_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(a.cos()));
  }
};
__name(_Sin, "Sin");
let Sin = _Sin;
registerOperation("sin", Sin);
const _cos_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.cos(a[x]);
  }
  return res;
}, "_cos_kernel");
function _cos_tensor(a, operation = null) {
  const kernel = _cos_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_cos_tensor, "_cos_tensor");
const _Cos = class _Cos extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _cos_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(a.sin().neg()));
  }
};
__name(_Cos, "Cos");
let Cos = _Cos;
registerOperation("cos", Cos);
const _tan_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.tan(a[x]);
  }
  return res;
}, "_tan_kernel");
function _tan_tensor(a, operation = null) {
  const kernel = _tan_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_tan_tensor, "_tan_tensor");
const _Tan = class _Tan extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _tan_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(a.cos().pow(-2)));
  }
};
__name(_Tan, "Tan");
let Tan = _Tan;
registerOperation("tan", Tan);
function _sum_tensor(a, operation = null) {
  return new Tensor(
    a.toArray().reduce((acc, val) => acc + val, 0),
    { requires_grad: a.requires_grad },
    { operation }
  );
}
__name(_sum_tensor, "_sum_tensor");
const _Sum = class _Sum extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _sum_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    const result = new Tensor(Array(a.dataLength()).fill(dz.item()));
    aFn.backward(result);
  }
};
__name(_Sum, "Sum");
let Sum = _Sum;
registerOperation("sum", Sum);
function _mean_tensor(a, operation = null) {
  return new Tensor(
    a.toArray().reduce((acc, val) => acc + val, 0) / a.dataLength(),
    { requires_grad: a.requires_grad },
    { operation }
  );
}
__name(_mean_tensor, "_mean_tensor");
const _Mean = class _Mean extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _mean_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    const result = new Tensor(Array(a.dataLength()).fill(dz.item() / a.dataLength()));
    aFn.backward(result);
  }
};
__name(_Mean, "Mean");
let Mean = _Mean;
registerOperation("mean", Mean);
function _transpose_tensor(a, dim0, dim1, operation = null) {
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
    { operation, shape: output_shape }
  );
}
__name(_transpose_tensor, "_transpose_tensor");
const _Transpose = class _Transpose extends Operation {
  dim0;
  dim1;
  _forward(a, dim0, dim1) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
      this.dim0 = dim0;
      this.dim1 = dim1;
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _transpose_tensor(a, dim0, dim1, this);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const dim0 = this.dim0;
    const dim1 = this.dim1;
    const [aFn] = this.next_functions;
    aFn.backward(dz.transpose(dim0, dim1));
  }
};
__name(_Transpose, "Transpose");
let Transpose = _Transpose;
registerOperation("transpose", Transpose);
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
  const dim_K = a_shape[a_shape.length - 1];
  for (let i = 0; i < output_size; i++) {
    const mn_idx = i % (dim_M * dim_N);
    const m = Math.floor(mn_idx / dim_N);
    const n = mn_idx % dim_N;
    let base_a = _get_original_index(padded_a_shape, broadcast_shape, i - n);
    let base_b = _get_original_index(padded_b_shape, broadcast_shape, i - m * dim_N);
    let sum2 = 0;
    for (let k = 0; k < dim_K; k++) {
      sum2 += a.data[base_a + k] * b.data[base_b + k * dim_N];
    }
    data[i] = sum2;
  }
  let shape_after_removing_extra_dims = [...broadcast_shape];
  if (a_1d) {
    shape_after_removing_extra_dims = shape_after_removing_extra_dims.slice(0, -2).concat([broadcast_shape[broadcast_shape.length - 1]]);
  }
  if (b_1d) {
    shape_after_removing_extra_dims = shape_after_removing_extra_dims.slice(0, -1);
  }
  return new Tensor(
    data,
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: shape_after_removing_extra_dims }
  );
}
__name(_matmul_tensor, "_matmul_tensor");
const _Matmul = class _Matmul extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _matmul_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(_Matmul, "Matmul");
let Matmul = _Matmul;
registerOperation("matmul", Matmul);
const _lt_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] < b[b_index] ? 1 : 0;
  }
  return res;
}, "_lt_kernel");
function _lt_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _lt_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_lt_tensor, "_lt_tensor");
const _Lt = class _Lt extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _lt_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(_Lt, "Lt");
let Lt = _Lt;
registerOperation("lt", Lt);
const _gt_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] > b[b_index] ? 1 : 0;
  }
  return res;
}, "_gt_kernel");
function _gt_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _gt_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_gt_tensor, "_gt_tensor");
const _Gt = class _Gt extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _gt_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(_Gt, "Gt");
let Gt = _Gt;
registerOperation("gt", Gt);
const _le_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] <= b[b_index] ? 1 : 0;
  }
  return res;
}, "_le_kernel");
function _le_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _le_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_le_tensor, "_le_tensor");
const _Le = class _Le extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _le_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(_Le, "Le");
let Le = _Le;
registerOperation("le", Le);
const _ge_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] >= b[b_index] ? 1 : 0;
  }
  return res;
}, "_ge_kernel");
function _ge_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _ge_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_ge_tensor, "_ge_tensor");
const _Ge = class _Ge extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _ge_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(_Ge, "Ge");
let Ge = _Ge;
registerOperation("ge", Ge);
const _eq_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] == b[b_index] ? 1 : 0;
  }
  return res;
}, "_eq_kernel");
function _eq_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _eq_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_eq_tensor, "_eq_tensor");
const _Eq = class _Eq extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _eq_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(_Eq, "Eq");
let Eq = _Eq;
registerOperation("eq", Eq);
const _ne_kernel = /* @__PURE__ */ __name(function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] != b[b_index] ? 1 : 0;
  }
  return res;
}, "_ne_kernel");
function _ne_tensor(a, b, operation = null) {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = _ne_kernel;
  const output_size = broadcast_shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape, output_size),
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation, shape: broadcast_shape }
  );
}
__name(_ne_tensor, "_ne_tensor");
const _Ne = class _Ne extends BinaryOperation {
  _forward(a, b) {
    if (a.requires_grad || b.requires_grad) {
      this.saved_tensors = [a, b];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
    return _ne_tensor(a, b, a.requires_grad || b.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
  }
};
__name(_Ne, "Ne");
let Ne = _Ne;
registerOperation("ne", Ne);
function get_shape_from_args(args) {
  if (Array.isArray(args[0])) {
    return args[0];
  }
  return args;
}
__name(get_shape_from_args, "get_shape_from_args");
function randn(...args) {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(Math.random()));
  tensor.shape = shape;
  return tensor;
}
__name(randn, "randn");
function rand(...args) {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(Math.random()));
  tensor.shape = shape;
  return tensor;
}
__name(rand, "rand");
function randint(low, high, shape) {
  const tensor = new Tensor(
    Array(shape.reduce((a, b) => a * b, 1)).fill(Math.floor(Math.random() * (high - low) + low))
  );
  tensor.shape = shape;
  return tensor;
}
__name(randint, "randint");
function ones(...args) {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(1));
  tensor.shape = shape;
  return tensor;
}
__name(ones, "ones");
function zeros(...args) {
  const shape = get_shape_from_args(args);
  const tensor = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(0));
  tensor.shape = shape;
  return tensor;
}
__name(zeros, "zeros");
function ones_like(tensor) {
  return ones(tensor.shape);
}
__name(ones_like, "ones_like");
function zeros_like(tensor) {
  return zeros(tensor.shape);
}
__name(zeros_like, "zeros_like");
function linspace(start, end, steps) {
  const data = [];
  const step = (end - start) / (steps - 1);
  for (let i = 0; i < steps - 1; i++) {
    data.push(start + i * step);
  }
  data.push(end);
  return new Tensor(data);
}
__name(linspace, "linspace");
function arange(start, end = void 0, step = 1) {
  const data = [];
  for (let i = start; i < end; i += step) {
    data.push(i);
  }
  return new Tensor(data);
}
__name(arange, "arange");
const _relu_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.max(a[x], 0);
  }
  return res;
}, "_relu_kernel");
function _relu_tensor(a, operation = null) {
  const kernel = _relu_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_relu_tensor, "_relu_tensor");
const _Relu = class _Relu extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _relu_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(a.gt(0)));
  }
};
__name(_Relu, "Relu");
let Relu = _Relu;
registerOperation("relu", Relu);
const _sigmoid_kernel = /* @__PURE__ */ __name(function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = 1 / (1 + Math.exp(-a[x]));
  }
  return res;
}, "_sigmoid_kernel");
function _sigmoid_tensor(a, operation = null) {
  const kernel = _sigmoid_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
__name(_sigmoid_tensor, "_sigmoid_tensor");
let Sigmoid$1 = (_a = class extends UnaryOperation {
  _forward(a) {
    if (a.requires_grad) {
      this.saved_tensors = [a];
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _sigmoid_tensor(a, a.requires_grad ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(a.exp().add(1).pow(-2).reciprocal().mul(a.exp()).mul(-1)));
  }
}, __name(_a, "Sigmoid"), _a);
registerOperation("sigmoid", Sigmoid$1);
const _Parameter = class _Parameter extends Tensor {
  constructor(data, options = {
    requires_grad: true
  }, internal_options = {}) {
    if (data instanceof Tensor) {
      super(data.data, { requires_grad: true }, { shape: data.shape });
    } else if (data instanceof _Parameter) {
      super(data.data, { requires_grad: true }, { shape: data.shape });
    } else {
      super(data, options, internal_options);
    }
  }
};
__name(_Parameter, "Parameter");
let Parameter = _Parameter;
const _Module = class _Module {
  _modules;
  _parameters;
  constructor() {
    this._parameters = {};
    this._modules = {};
  }
  register_parameter(parameter_name, parameter) {
    this._parameters[parameter_name] = parameter;
  }
  register_module(module_name, module) {
    this._modules[module_name] = module;
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
    for (const module of Object.values(this._modules)) {
      params = params.concat(module.parameters());
    }
    return params;
  }
};
__name(_Module, "Module");
let Module = _Module;
const _Linear = class _Linear extends Module {
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
};
__name(_Linear, "Linear");
let Linear = _Linear;
const _ReLU = class _ReLU extends Module {
  constructor() {
    super();
  }
  forward(input) {
    return relu(input);
  }
};
__name(_ReLU, "ReLU");
let ReLU = _ReLU;
const _Sigmoid = class _Sigmoid extends Module {
  constructor() {
    super();
  }
  forward(input) {
    return sigmoid(input);
  }
};
__name(_Sigmoid, "Sigmoid");
let Sigmoid = _Sigmoid;
const _Loss = class _Loss {
};
__name(_Loss, "Loss");
let Loss = _Loss;
const _MSELoss = class _MSELoss extends Loss {
  constructor() {
    super();
  }
  forward(input, target) {
    return input.sub(target).pow(2).mean();
  }
};
__name(_MSELoss, "MSELoss");
let MSELoss = _MSELoss;
const _L1Loss = class _L1Loss extends Loss {
  constructor() {
    super();
  }
  forward(input, target) {
    return input.sub(target).abs().mean();
  }
};
__name(_L1Loss, "L1Loss");
let L1Loss = _L1Loss;
const _BCELoss = class _BCELoss extends Loss {
  weight;
  constructor(weight = null) {
    super();
    this.weight = weight;
  }
  forward(input, target) {
    const left = target.mul(input.log());
    const right = target.neg().add(1).mul(input.neg().add(1).log());
    const loss = left.add(right).neg().mean();
    if (this.weight) {
      return loss.mul(this.weight);
    }
    return loss;
  }
};
__name(_BCELoss, "BCELoss");
let BCELoss = _BCELoss;
function generate_function(opname) {
  return (...args) => {
    const operation = new (getOperation(opname))();
    return operation.forward(...args);
  };
}
__name(generate_function, "generate_function");
function generate_unary_function(opname) {
  return (a) => {
    if (typeof a == "number") {
      a = new Tensor(a);
    }
    const operation = new (getOperation(opname))();
    return operation.forward(a);
  };
}
__name(generate_unary_function, "generate_unary_function");
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
__name(generate_binary_function, "generate_binary_function");
const relu = generate_unary_function("relu");
const sigmoid = generate_unary_function("sigmoid");
const functional$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu,
  sigmoid
}, Symbol.toStringTag, { value: "Module" }));
const index$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss,
  L1Loss,
  Linear,
  MSELoss,
  Module,
  Parameter,
  ReLU,
  Sigmoid,
  functional: functional$1
}, Symbol.toStringTag, { value: "Module" }));
const _Optimizer = class _Optimizer {
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
};
__name(_Optimizer, "Optimizer");
let Optimizer = _Optimizer;
const _SGD = class _SGD extends Optimizer {
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
      if (this.weight_decay !== 0) {
        g = g.add(param.mul(this.weight_decay));
      }
      if (this.momentum !== 0) {
        if (this.state.has(param)) {
          let buf2 = this.state.get(param).velocity;
          buf2 = buf2.mul(this.momentum);
          buf2 = buf2.add(g.mul(1 - this.dampening));
          this.state.set(param, { velocity: buf2 });
        } else {
          this.state.set(param, { velocity: g });
        }
        let buf = this.state.get(param).velocity;
        if (this.nesterov) {
          g = g.add(buf.mul(this.momentum));
        } else {
          g = buf;
        }
        this.state.set(param, { velocity: buf });
      }
      const newParam = param.sub(g.mul(this.lr));
      param.data = newParam.data;
    }
  }
};
__name(_SGD, "SGD");
let SGD = _SGD;
const _Adam = class _Adam extends Optimizer {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(params, lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weight_decay = 0, amsgrad = false, maximize = false) {
    super(params, {});
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.weight_decay = weight_decay;
    this.amsgrad = amsgrad;
    this.maximize = maximize;
  }
  step() {
    this.step_count += 1;
    for (const param of this.params) {
      let grad = this.maximize ? param.grad.mul(-1) : param.grad;
      if (this.weight_decay !== 0) {
        grad = grad.add(param.mul(this.weight_decay));
      }
      if (!this.state.has(param)) {
        this.state.set(param, {
          m: zeros_like(param),
          v: zeros_like(param),
          vmax: zeros_like(param)
        });
      }
      const state = this.state.get(param);
      state.m = state.m.mul(this.beta1).add(grad.mul(1 - this.beta1));
      state.v = state.v.mul(this.beta2).add(grad.mul(grad).mul(1 - this.beta2));
      const biasCorrection1 = 1 - Math.pow(this.beta1, this.step_count);
      const biasCorrection2 = 1 - Math.pow(this.beta2, this.step_count);
      let vhat;
      const mhat = state.m.div(biasCorrection1);
      if (this.amsgrad) {
        state.vmax = state.vmax.maximum(state.v);
        vhat = state.vmax.div(biasCorrection2);
      } else {
        vhat = state.v.div(biasCorrection2);
      }
      const update = mhat.div(vhat.sqrt().add(this.eps)).mul(this.lr);
      const newParam = param.sub(update);
      param.data = newParam.data;
    }
  }
};
__name(_Adam, "Adam");
let Adam = _Adam;
const index = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam,
  Optimizer,
  SGD
}, Symbol.toStringTag, { value: "Module" }));
export {
  Abs,
  Add,
  Cos,
  Div,
  Eq,
  Exp,
  Fmod,
  Ge,
  Gt,
  Le,
  Log,
  Lt,
  Matmul,
  Maximum,
  Mean,
  Minimum,
  Mul,
  Ne,
  Neg,
  Operation,
  Pow,
  PowInt,
  Reciprocal,
  Reshape,
  Sign,
  Sin,
  Sqrt,
  Square,
  Sub,
  Sum,
  Tan,
  Tensor,
  Transpose,
  Unsqueeze,
  __Left_index__,
  __Right_index__,
  __left_index__,
  __right_index__,
  abs,
  add,
  arange,
  cos,
  div,
  eq,
  eventBus,
  exp,
  fmod,
  ge,
  gt,
  le,
  linspace,
  log,
  lt,
  matmul,
  maximum,
  mean,
  minimum,
  mul,
  ne,
  neg,
  index$1 as nn,
  ones,
  ones_like,
  index as optim,
  pow,
  rand,
  randint,
  randn,
  reciprocal,
  reshape,
  sign,
  sin,
  sqrt,
  square,
  sub,
  sum,
  tan,
  transpose,
  unsqueeze,
  zeros,
  zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
