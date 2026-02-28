"use strict";
Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
let globalId = 0;
const getNextId = () => {
  return globalId++;
};
const eventBus = new EventTarget();
const events = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_ACCUMULATE_GRAD: "operation.accumulateGrad"
};
class Operation {
  id = getNextId();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...args) {
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        args
      }
    }));
    const result = this._forward(...args);
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_AFTER_FORWARD, {
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
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz } }));
    for (const x of this._retained_tensors) {
      if (!x.grad) {
        x.grad = new Tensor(new Array(x.dataLength()).fill(0));
      }
      x.grad = x.grad.add(dz);
    }
    this._backward(dz);
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz } }));
  }
}
class NullOp extends Operation {
  _forward(...args) {
    throw new Error("NullOp should not be called");
  }
  _backward(dz) {
    return;
  }
}
const nullOp = new NullOp();
class UnaryOperation extends Operation {
}
class BinaryOperation extends Operation {
}
class AccumulateGrad extends UnaryOperation {
  variable;
  _forward(variable) {
    this.variable = variable;
    return variable;
  }
  _backward(dz) {
    if (!this.variable.grad) {
      this.variable.grad = new Tensor(new Array(this.variable.dataLength()).fill(0));
    }
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_ACCUMULATE_GRAD, { detail: { operation: this, dz } }));
    this.variable.grad = this.variable.grad.add(dz);
  }
}
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
      grad = new Tensor(1);
    } else {
      grad.toArray_();
    }
    if (this.grad_fn) {
      eventBus.dispatchEvent(new CustomEvent(events.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } }));
      this.grad_fn.backward(grad);
      eventBus.dispatchEvent(new CustomEvent(events.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } }));
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
}
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
const __left_index__ = generate_binary_function("__left_index__");
const __right_index__ = generate_binary_function("__right_index__");
const add = generate_binary_function("add");
const sub = generate_binary_function("sub");
const mul = generate_binary_function("mul");
const div = generate_binary_function("div");
const pow = generate_binary_function("pow");
const fmod = generate_binary_function("fmod");
const maximum = generate_binary_function("maximum");
const minimum = generate_binary_function("minimum");
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
const ___left_index___kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    _get_original_index(bs, bcs, x);
    res[x] = a_index;
  }
  return res;
};
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
class __Left_index__ extends BinaryOperation {
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
}
registerOperation("__left_index__", __Left_index__);
const ___right_index___kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = b_index;
  }
  return res;
};
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
class __Right_index__ extends BinaryOperation {
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
}
registerOperation("__right_index__", __Right_index__);
const _add_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] + b[b_index];
  }
  return res;
};
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
class Add extends BinaryOperation {
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
}
registerOperation("add", Add);
const _sub_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] - b[b_index];
  }
  return res;
};
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
class Sub extends BinaryOperation {
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
}
registerOperation("sub", Sub);
const _mul_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] * b[b_index];
  }
  return res;
};
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
class Mul extends BinaryOperation {
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
}
registerOperation("mul", Mul);
const _div_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] / b[b_index];
  }
  return res;
};
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
class Div extends BinaryOperation {
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
}
registerOperation("div", Div);
const _pow_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = Math.pow(a[a_index], b[b_index]);
  }
  return res;
};
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
class Pow extends BinaryOperation {
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
}
registerOperation("pow", Pow);
const _fmod_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] % b[b_index];
  }
  return res;
};
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
class Fmod extends BinaryOperation {
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
}
registerOperation("fmod", Fmod);
const _maximum_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = Math.max(a[a_index], b[b_index]);
  }
  return res;
};
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
class Maximum extends BinaryOperation {
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
}
registerOperation("maximum", Maximum);
const _minimum_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = Math.min(a[a_index], b[b_index]);
  }
  return res;
};
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
class Minimum extends BinaryOperation {
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
}
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
class PowInt extends Operation {
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
}
registerOperation("powint", PowInt);
const _log_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.log(a[x]);
  }
  return res;
};
function _log_tensor(a, operation = null) {
  const kernel = _log_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Log extends UnaryOperation {
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
}
registerOperation("log", Log);
const _sqrt_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.sqrt(a[x]);
  }
  return res;
};
function _sqrt_tensor(a, operation = null) {
  const kernel = _sqrt_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Sqrt extends UnaryOperation {
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
}
registerOperation("sqrt", Sqrt);
const _exp_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.exp(a[x]);
  }
  return res;
};
function _exp_tensor(a, operation = null) {
  const kernel = _exp_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Exp extends UnaryOperation {
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
}
registerOperation("exp", Exp);
const _square_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = a[x] * a[x];
  }
  return res;
};
function _square_tensor(a, operation = null) {
  const kernel = _square_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Square extends UnaryOperation {
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
}
registerOperation("square", Square);
const _abs_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.abs(a[x]);
  }
  return res;
};
function _abs_tensor(a, operation = null) {
  const kernel = _abs_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Abs extends UnaryOperation {
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
}
registerOperation("abs", Abs);
const _sign_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.sign(a[x]);
  }
  return res;
};
function _sign_tensor(a, operation = null) {
  const kernel = _sign_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Sign extends UnaryOperation {
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
}
registerOperation("sign", Sign);
const _neg_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = -a[x];
  }
  return res;
};
function _neg_tensor(a, operation = null) {
  const kernel = _neg_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Neg extends UnaryOperation {
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
}
registerOperation("neg", Neg);
const _reciprocal_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = 1 / a[x];
  }
  return res;
};
function _reciprocal_tensor(a, operation = null) {
  const kernel = _reciprocal_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Reciprocal extends UnaryOperation {
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
}
registerOperation("reciprocal", Reciprocal);
class Reshape extends Operation {
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
}
registerOperation("reshape", Reshape);
class Unsqueeze extends Operation {
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
}
registerOperation("unsqueeze", Unsqueeze);
const _sin_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.sin(a[x]);
  }
  return res;
};
function _sin_tensor(a, operation = null) {
  const kernel = _sin_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Sin extends UnaryOperation {
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
}
registerOperation("sin", Sin);
const _cos_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.cos(a[x]);
  }
  return res;
};
function _cos_tensor(a, operation = null) {
  const kernel = _cos_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Cos extends UnaryOperation {
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
}
registerOperation("cos", Cos);
const _tan_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.tan(a[x]);
  }
  return res;
};
function _tan_tensor(a, operation = null) {
  const kernel = _tan_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Tan extends UnaryOperation {
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
}
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
class Transpose extends Operation {
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
}
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
class Matmul extends BinaryOperation {
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
}
registerOperation("matmul", Matmul);
const _lt_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] < b[b_index] ? 1 : 0;
  }
  return res;
};
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
class Lt extends BinaryOperation {
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
}
registerOperation("lt", Lt);
const _gt_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] > b[b_index] ? 1 : 0;
  }
  return res;
};
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
class Gt extends BinaryOperation {
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
}
registerOperation("gt", Gt);
const _le_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] <= b[b_index] ? 1 : 0;
  }
  return res;
};
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
class Le extends BinaryOperation {
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
}
registerOperation("le", Le);
const _ge_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] >= b[b_index] ? 1 : 0;
  }
  return res;
};
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
class Ge extends BinaryOperation {
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
}
registerOperation("ge", Ge);
const _eq_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] == b[b_index] ? 1 : 0;
  }
  return res;
};
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
class Eq extends BinaryOperation {
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
}
registerOperation("eq", Eq);
const _ne_kernel = function(a, as, b, bs, bcs, output_size) {
  const res = Array(output_size);
  for (let x = 0; x < output_size; x++) {
    const a_index = _get_original_index(as, bcs, x);
    const b_index = _get_original_index(bs, bcs, x);
    res[x] = a[a_index] != b[b_index] ? 1 : 0;
  }
  return res;
};
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
class Ne extends BinaryOperation {
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
function ones_like(tensor) {
  return ones(tensor.shape);
}
function zeros_like(tensor) {
  return zeros(tensor.shape);
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
const _relu_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = Math.max(a[x], 0);
  }
  return res;
};
function _relu_tensor(a, operation = null) {
  const kernel = _relu_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
class Relu extends UnaryOperation {
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
}
registerOperation("relu", Relu);
const _sigmoid_kernel = function(a, output) {
  const res = new Array(output);
  for (let x = 0; x < output; x++) {
    res[x] = 1 / (1 + Math.exp(-a[x]));
  }
  return res;
};
function _sigmoid_tensor(a, operation = null) {
  const kernel = _sigmoid_kernel;
  const output = a.shape.reduce((acc, val) => acc * val, 1);
  return new Tensor(
    kernel(a.data, output),
    { requires_grad: a.requires_grad },
    { operation, shape: a.shape }
  );
}
let Sigmoid$1 = class Sigmoid extends UnaryOperation {
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
};
registerOperation("sigmoid", Sigmoid$1);
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
class Sigmoid2 extends Module {
  constructor() {
    super();
  }
  forward(input) {
    return sigmoid(input);
  }
}
class Loss {
}
class MSELoss extends Loss {
  constructor() {
    super();
  }
  forward(input, target) {
    return input.sub(target).pow(2).mean();
  }
}
class L1Loss extends Loss {
  constructor() {
    super();
  }
  forward(input, target) {
    return input.sub(target).abs().mean();
  }
}
class BCELoss extends Loss {
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
const sigmoid = generate_unary_function("sigmoid");
const functional = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
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
  Sigmoid: Sigmoid2,
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
}
class Adam extends Optimizer {
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
}
const index = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam,
  Optimizer,
  SGD
}, Symbol.toStringTag, { value: "Module" }));
exports.Abs = Abs;
exports.AccumulateGrad = AccumulateGrad;
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
exports.Maximum = Maximum;
exports.Mean = Mean;
exports.Minimum = Minimum;
exports.Mul = Mul;
exports.Ne = Ne;
exports.Neg = Neg;
exports.Operation = Operation;
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
exports.__Left_index__ = __Left_index__;
exports.__Right_index__ = __Right_index__;
exports.__left_index__ = __left_index__;
exports.__right_index__ = __right_index__;
exports.abs = abs;
exports.add = add;
exports.arange = arange;
exports.cos = cos;
exports.div = div;
exports.eq = eq;
exports.eventBus = eventBus;
exports.events = events;
exports.exp = exp;
exports.fmod = fmod;
exports.ge = ge;
exports.gt = gt;
exports.le = le;
exports.linspace = linspace;
exports.log = log;
exports.lt = lt;
exports.matmul = matmul;
exports.maximum = maximum;
exports.mean = mean;
exports.minimum = minimum;
exports.mul = mul;
exports.ne = ne;
exports.neg = neg;
exports.nn = index$1;
exports.ones = ones;
exports.ones_like = ones_like;
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
exports.zeros_like = zeros_like;
