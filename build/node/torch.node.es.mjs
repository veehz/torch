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
function _unbroadcast(result_shape, original_shape, result) {
  const this_shape = _pad_shape(original_shape, result_shape);
  const unbroadcasted_result = new Array(original_shape.reduce((acc, cur) => acc * cur, 1)).fill(0);
  for (let i = 0; i < result.length; i++) {
    unbroadcasted_result[_get_original_index(this_shape, result_shape, i)] += result[i];
  }
  return unbroadcasted_result;
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
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function _numel(shape) {
  return shape.reduce((a, b) => a * b, 1);
}
function _get_shape_from_args(args) {
  if (Array.isArray(args[0])) {
    return args[0];
  }
  return args;
}
let _rng = () => Math.random();
function getRng() {
  return _rng;
}
function manual_seed(seed2) {
  seed2 = seed2 >>> 0;
  _rng = mulberry32(seed2);
  return seed2;
}
function seed() {
  const s = Math.random() * 4294967295 >>> 0;
  _rng = mulberry32(s);
  return s;
}
function mulberry32(seed2) {
  return function() {
    let t = seed2 += 1831565813;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function uniformDist(min2 = 0, max2 = 1) {
  return () => min2 + getRng()() * (max2 - min2);
}
function normalDist(mean2 = 0, std = 1) {
  return function() {
    const u = 1 - getRng()();
    const v = getRng()();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return z * std + mean2;
  };
}
function randn(...args) {
  const shape = _get_shape_from_args(args);
  const tensor2 = new Tensor(Array.from({ length: _numel(shape) }, normalDist()));
  tensor2.shape = shape;
  return tensor2;
}
function rand(...args) {
  const shape = _get_shape_from_args(args);
  const tensor2 = new Tensor(Array.from({ length: _numel(shape) }, uniformDist()));
  tensor2.shape = shape;
  return tensor2;
}
function randint(low, high, shape) {
  const tensor2 = new Tensor(
    Array.from({ length: _numel(shape) }, () => Math.floor(uniformDist(low, high)()))
  );
  tensor2.shape = shape;
  return tensor2;
}
function randperm(n) {
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < n; i++) {
    const j = Math.floor(uniformDist()() * (n - i)) + i;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  const tensor2 = new Tensor(arr);
  return tensor2;
}
function tensor(data, requires_grad = false) {
  return new Tensor(data, { requires_grad });
}
function ones(...args) {
  const shape = _get_shape_from_args(args);
  const tensor2 = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(1));
  tensor2.shape = shape;
  return tensor2;
}
function zeros(...args) {
  const shape = _get_shape_from_args(args);
  const tensor2 = new Tensor(Array(shape.reduce((a, b) => a * b, 1)).fill(0));
  tensor2.shape = shape;
  return tensor2;
}
function ones_like(tensor2) {
  return ones(tensor2.shape);
}
function zeros_like(tensor2) {
  return zeros(tensor2.shape);
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
let _grad_enabled = true;
function is_grad_enabled() {
  return _grad_enabled;
}
function enable_no_grad() {
  const prev = _grad_enabled;
  _grad_enabled = false;
  return prev;
}
function disable_no_grad(prev) {
  _grad_enabled = prev;
}
function no_grad(fn) {
  const prev = enable_no_grad();
  try {
    return fn();
  } finally {
    disable_no_grad(prev);
  }
}
function resultRequiresGrad(...args) {
  if (!is_grad_enabled()) return false;
  for (const arg of args) {
    if (arg instanceof Tensor && arg.requires_grad) {
      return true;
    }
  }
  return false;
}
class TorchFunction {
  id = getNextId();
  opName = "";
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...args) {
    const requires_grad = resultRequiresGrad(...args);
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad,
        args
      }
    }));
    const result = this._forward(...args);
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad,
        args,
        result
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
class NullOp extends TorchFunction {
  _forward(..._args) {
    throw new Error("NullOp should not be called");
  }
  _backward(_dz) {
    return;
  }
}
const nullOp = new NullOp();
class UnaryFunction extends TorchFunction {
}
class BinaryFunction extends TorchFunction {
}
class AccumulateGrad extends UnaryFunction {
  variable;
  _forward(variable) {
    this.variable = variable;
    return variable;
  }
  _backward(dz) {
    if (!this.variable.grad) {
      this.variable.grad = zeros_like(this.variable);
    }
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz } }));
    if (typeof dz === "number") {
      this.variable.grad = this.variable.grad.add(dz);
    } else {
      const unbroadcasted_dz = _unbroadcast(dz.shape, this.variable.shape, dz.data);
      this.variable.grad = this.variable.grad.add(new Tensor(unbroadcasted_dz, {}, { shape: this.variable.shape }));
    }
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz } }));
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
    const op = new (getOperation(name))();
    op.opName = name;
    operations_cache.set(name, op);
    return op;
  }
  return operation;
}
function createOperation(name) {
  const op = new (getOperation(name))();
  op.opName = name;
  return op;
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
function _assert_shape(data, shape) {
  if (Array.isArray(data)) {
    if (data.length !== shape[0]) {
      throw new Error(
        `Shape mismatch at dim ${shape.length}: expected ${shape[0]}, got ${data.length}`
      );
    }
    for (let i = 0; i < data.length; i++) {
      _assert_shape(data[i], shape.slice(1));
    }
  } else if (ArrayBuffer.isView(data)) {
    if (shape.length !== 1) {
      throw new Error(`Shape mismatch at dim ${shape.length}: expected 1D, got ${shape}`);
    }
    if (data.length !== shape[0]) {
      throw new Error(
        `Shape mismatch at dim ${shape.length}: expected ${shape[0]}, got ${data.length}`
      );
    }
  } else {
    if (shape.length !== 0) {
      throw new Error(`Shape mismatch at dim ${shape.length}: expected scalar, got ${data}`);
    }
  }
}
function _get_and_assert_shape(data) {
  const shape = _get_shape(data);
  _assert_shape(data, shape);
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
  // Auto-generated ID
  id = getNextId();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(data, options = {}, internal_options = {}) {
    this.data = _flatten(data);
    this.requires_grad = options.requires_grad ?? false;
    if (options.name) {
      this.name = options.name;
    }
    this.shape = internal_options.shape ?? _get_and_assert_shape(data);
    this.grad_fn = internal_options.operation ?? null;
    if (this.requires_grad && !this.grad_fn) {
      const acc = new AccumulateGrad();
      acc.variable = this;
      this.grad_fn = acc;
    }
  }
  size(dim) {
    if (dim !== void 0) {
      if (dim < 0) {
        dim += this.shape.length;
      }
      if (dim < 0 || dim >= this.shape.length) {
        throw new Error(
          `Dimension out of range (expected to be in range of [${-this.shape.length}, ${this.shape.length - 1}], but got ${dim})`
        );
      }
      return this.shape[dim];
    }
    return this.shape;
  }
  toArray_() {
    return;
  }
  toFlatArray() {
    return this.data;
  }
  toArray() {
    if (this.shape.length === 0) {
      return this.data[0];
    }
    let flatIndex = 0;
    const flatData = this.data;
    const buildDimension = (currentDim) => {
      const size = this.shape[currentDim];
      const result = new Array(size);
      const isLastDimension = currentDim === this.shape.length - 1;
      for (let i = 0; i < size; i++) {
        if (isLastDimension) {
          result[i] = flatData[flatIndex++];
        } else {
          result[i] = buildDimension(currentDim + 1);
        }
      }
      return result;
    };
    return buildDimension(0);
  }
  toString() {
    let extra = "";
    if (this.name) {
      extra += `, name="${this.name}"`;
    }
    if (this.dataLength() == 0 && this.shape.length > 0) {
      extra += `, size=(${this.shape.join(", ")})`;
    }
    if (this.requires_grad) {
      extra += ", requires_grad=True";
    }
    return `Tensor(${JSON.stringify(this.toArray())}${extra})`;
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(opName) {
    const operation = resultRequiresGrad(this) ? createOperation(opName) : getOperationCache(opName);
    return operation.forward(this);
  }
  _executeBinaryOp(opName, other) {
    if (typeof other == "number") {
      other = new Tensor(other);
    }
    const operation = resultRequiresGrad(this, other) ? createOperation(opName) : getOperationCache(opName);
    return operation.forward(this, other);
  }
  _executeOpRaw(opName, ...args) {
    const operation = createOperation(opName);
    return operation.forward(this, ...args);
  }
  item() {
    if (this.dataLength() !== 1) {
      throw new Error("Tensor.item() is only valid for scalars");
    }
    return this.data[0];
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
      eventBus.dispatchEvent(
        new CustomEvent(events.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })
      );
      this.grad_fn.backward(grad);
      eventBus.dispatchEvent(
        new CustomEvent(events.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })
      );
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
  nan_to_num() {
    return this._executeUnaryOp("nan_to_num");
  }
  reshape(shape) {
    return this._executeOpRaw("reshape", shape);
  }
  squeeze(dim) {
    return this._executeOpRaw("squeeze", dim);
  }
  unsqueeze(dim) {
    return this._executeOpRaw("unsqueeze", dim);
  }
  expand(sizes) {
    return this._executeOpRaw("expand", sizes);
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
  sum(dim, keepdim = false) {
    return this._executeOpRaw("sum", dim, keepdim);
  }
  mean(dim, keepdim = false) {
    return this._executeOpRaw("mean", dim, keepdim);
  }
  max(dim, keepdim = false) {
    return this._executeOpRaw("max", dim, keepdim);
  }
  min(dim, keepdim = false) {
    return this._executeOpRaw("min", dim, keepdim);
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
  allclose(other, rtol = 1e-5, atol = 1e-8, equal_nan = false) {
    if (this.data.length !== other.data.length) return false;
    for (let i = 0; i < this.data.length; i++) {
      const av = this.data[i], bv = other.data[i];
      if (equal_nan && isNaN(av) && isNaN(bv)) continue;
      if (isNaN(av) || isNaN(bv)) return false;
      if (Math.abs(av - bv) > atol + rtol * Math.abs(bv)) return false;
    }
    return true;
  }
  numel() {
    return this.dataLength();
  }
  // other
  sigmoid() {
    return this._executeUnaryOp("sigmoid");
  }
}
function generate_function$1(opname) {
  return (...args) => {
    const operation = createOperation(opname);
    return operation.forward(...args);
  };
}
function generate_unary_function$1(opname) {
  return (a) => {
    if (typeof a == "number") {
      a = new Tensor(a);
    }
    const operation = createOperation(opname);
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
    const operation = createOperation(opname);
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
const nan_to_num = generate_unary_function$1("nan_to_num");
const reshape = generate_function$1("reshape");
const squeeze = generate_function$1("squeeze");
const unsqueeze = generate_function$1("unsqueeze");
const expand = generate_function$1("expand");
const sin = generate_unary_function$1("sin");
const cos = generate_unary_function$1("cos");
const tan = generate_unary_function$1("tan");
const sum = generate_function$1("sum");
const mean = generate_function$1("mean");
const min = generate_function$1("min");
const max = generate_function$1("max");
const transpose = generate_function$1("transpose");
const matmul = generate_binary_function("matmul");
const lt = generate_binary_function("lt");
const gt = generate_binary_function("gt");
const le = generate_binary_function("le");
const ge = generate_binary_function("ge");
const eq = generate_binary_function("eq");
const ne = generate_binary_function("ne");
function allclose(a, b, rtol = 1e-5, atol = 1e-8, equal_nan = false) {
  return a.allclose(b, rtol, atol, equal_nan);
}
function numel(a) {
  return a.dataLength();
}
function _get_strides(shape) {
  const strides = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}
function _unravel_index(index2, strides) {
  return strides.map((stride) => {
    const coord = Math.floor(index2 / stride);
    index2 %= stride;
    return coord;
  });
}
function _ravel_index(coords, strides) {
  return coords.reduce((acc, coord, i) => acc + coord * strides[i], 0);
}
function _get_reduction_shape(shape, dim, keepdim = false) {
  if (dim === void 0) return keepdim ? shape.map(() => 1) : [];
  const dims = Array.isArray(dim) ? dim : [dim];
  const normalized_dims = dims.map((d) => d < 0 ? d + shape.length : d);
  if (keepdim) {
    return shape.map((s, i) => normalized_dims.includes(i) ? 1 : s);
  } else {
    return shape.filter((_, i) => !normalized_dims.includes(i));
  }
}
function BinaryFunctionMixin(operation, backward_operations, opName = null) {
  const kernel = (a, as, b, bs, bcs, output_size) => {
    const res = Array(output_size);
    for (let x = 0; x < output_size; x++) {
      const a_index = _get_original_index(as, bcs, x);
      const b_index = _get_original_index(bs, bcs, x);
      res[x] = operation(a, b, a_index, b_index);
    }
    return res;
  };
  const forward_tensor = (a, b, operation2 = null) => {
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
      ),
      { requires_grad: resultRequiresGrad(a, b) },
      { operation: operation2, shape: broadcast_shape }
    );
  };
  const result = {
    [opName]: class extends BinaryFunction {
      _forward(a, b) {
        const rg = resultRequiresGrad(a, b);
        if (rg) {
          this.saved_tensors = [a, b];
        }
        this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
        this.next_functions.push(b.grad_fn ? b.grad_fn : nullOp);
        return forward_tensor(a, b, rg ? this : null);
      }
      _backward(dz) {
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
function UnaryFunctionMixin(operation, backward_operations, opName = null) {
  const kernel = (a, output_size) => {
    const res = Array(output_size);
    for (let x = 0; x < output_size; x++) {
      res[x] = operation(a, x);
    }
    return res;
  };
  const forward_tensor = (a, operation2 = null) => {
    const output_size = a.dataLength();
    return new Tensor(
      kernel(a.data, output_size),
      { requires_grad: resultRequiresGrad(a) },
      { operation: operation2, shape: a.shape }
    );
  };
  const result = {
    [opName]: class extends UnaryFunction {
      _forward(a) {
        const rg = resultRequiresGrad(a);
        if (rg) {
          this.saved_tensors = [a];
        }
        this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
        return forward_tensor(a, rg ? this : null);
      }
      _backward(dz) {
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
function ReductionFunctionMixin(init_val, reduce_op, backward_operations, opName = null, finalize_op) {
  const result = {
    [opName]: class extends TorchFunction {
      dim;
      keepdim;
      _forward(a, dim, keepdim = false) {
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
        const counts = new Array(out_size).fill(0);
        const in_strides = _get_strides(a.shape);
        const out_strides = _get_strides(out_shape);
        const dims = dim === void 0 ? [] : Array.isArray(dim) ? dim : [dim];
        const normalized_dims = dims.map((d) => d < 0 ? d + a.shape.length : d);
        const is_full_reduce = dim === void 0;
        for (let i = 0; i < a.data.length; i++) {
          const in_coords = _unravel_index(i, in_strides);
          let out_coords;
          if (is_full_reduce) {
            out_coords = keepdim ? in_coords.map(() => 0) : [];
          } else {
            out_coords = [];
            for (let j = 0; j < a.shape.length; j++) {
              if (normalized_dims.includes(j)) {
                if (keepdim) out_coords.push(0);
              } else {
                out_coords.push(in_coords[j]);
              }
            }
          }
          const out_idx = _ravel_index(out_coords, out_strides);
          res_data[out_idx] = reduce_op(res_data[out_idx], a.data[i]);
          counts[out_idx]++;
        }
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
      _backward(dz) {
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
function unbroadcast(result, original_shape) {
  const unbroadcasted_result = _unbroadcast(result.shape, original_shape, result.data);
  return new Tensor(unbroadcasted_result, { requires_grad: result.requires_grad }, { shape: original_shape });
}
function broadcast(tensor2, result_shape) {
  return tensor2.mul(ones(result_shape));
}
BinaryFunctionMixin(
  (a, b, a_index, _b_index) => a_index,
  () => {
  },
  "__left_index__"
);
BinaryFunctionMixin(
  (a, b, _a_index, b_index) => b_index,
  () => {
  },
  "__right_index__"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] + b[b_index],
  (_a, _b, aFn, bFn, dz) => {
    aFn.backward(dz);
    bFn.backward(dz);
  },
  "add"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] - b[b_index],
  (_a, _b, aFn, bFn, dz) => {
    aFn.backward(dz);
    bFn.backward(dz.mul(new Tensor(-1)));
  },
  "sub"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] * b[b_index],
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz.mul(b));
    bFn.backward(dz.mul(a));
  },
  "mul"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] / b[b_index],
  (a, b, aFn, bFn, dz) => {
    aFn.backward(dz.div(b));
    bFn.backward(dz.mul(a).mul(new Tensor(-1)).div(b).div(b));
  },
  "div"
);
function _where(mask, x, fallback) {
  const fb = typeof fallback === "number" ? fallback : null;
  const data = new Array(x.dataLength());
  for (let i = 0; i < data.length; i++) {
    data[i] = mask.data[i] ? x.data[i] : fb !== null ? fb : fallback.data[i];
  }
  return new Tensor(data, {}, { shape: x.shape });
}
BinaryFunctionMixin(
  (a, b, a_index, b_index) => Math.pow(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    const ga = dz.mul(b).mul(a.pow(b.sub(new Tensor(1))));
    const gb = dz.mul(a.pow(b)).mul(a.log());
    aFn.backward(_where(a.ne(0), ga, ga.nan_to_num()));
    bFn.backward(_where(a.ne(0), gb, 0));
  },
  "pow"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] % b[b_index],
  (_a, _b, aFn, _bFn, dz) => {
    aFn.backward(dz);
  },
  "fmod"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => Math.max(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    const eq_mask = a.eq(b);
    const a_mask = a.gt(b).add(eq_mask.mul(new Tensor(0.5)));
    const b_mask = b.gt(a).add(eq_mask.mul(new Tensor(0.5)));
    aFn.backward(dz.mul(a_mask));
    bFn.backward(dz.mul(b_mask));
  },
  "maximum"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => Math.min(a[a_index], b[b_index]),
  (a, b, aFn, bFn, dz) => {
    const eq_mask = a.eq(b);
    const a_mask = a.lt(b).add(eq_mask.mul(new Tensor(0.5)));
    const b_mask = b.lt(a).add(eq_mask.mul(new Tensor(0.5)));
    aFn.backward(dz.mul(a_mask));
    bFn.backward(dz.mul(b_mask));
  },
  "minimum"
);
function _powint_tensor(a, n, operation = null) {
  const data = new Array(a.dataLength());
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.pow(a.data[i], n);
  }
  return new Tensor(
    data,
    { requires_grad: resultRequiresGrad(a) },
    { operation, shape: a.shape }
  );
}
class PowInt extends TorchFunction {
  n;
  _forward(a, n) {
    const rg = resultRequiresGrad(a);
    if (rg) {
      this.saved_tensors = [a];
      this.n = n;
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _powint_tensor(a, n, rg ? this : null);
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const n = this.n;
    const [aFn] = this.next_functions;
    aFn.backward(dz.mul(n).mul(a.pow(n - 1)));
  }
}
registerOperation("powint", PowInt);
UnaryFunctionMixin(
  (a, a_index) => Math.log(a[a_index]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(new Tensor(1).div(a)));
  },
  "log"
);
UnaryFunctionMixin(
  (a, x) => Math.sqrt(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(new Tensor(1).div(a.sqrt()).div(2)));
  },
  "sqrt"
);
UnaryFunctionMixin(
  (a, x) => Math.exp(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.exp()));
  },
  "exp"
);
UnaryFunctionMixin(
  (a, x) => a[x] * a[x],
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a).mul(new Tensor(2)));
  },
  "square"
);
UnaryFunctionMixin(
  (a, x) => Math.abs(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(sign(a)));
  },
  "abs"
);
UnaryFunctionMixin(
  (a, x) => Math.sign(a[x]),
  (_a, aFn) => {
    aFn.backward(0);
  },
  "sign"
);
UnaryFunctionMixin(
  (a, x) => -a[x],
  (_a, aFn, dz) => {
    aFn.backward(dz.mul(new Tensor(-1)));
  },
  "neg"
);
UnaryFunctionMixin(
  (a, x) => 1 / a[x],
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.pow(-2)).neg());
  },
  "reciprocal"
);
UnaryFunctionMixin(
  (a, x) => {
    const v = a[x];
    if (Number.isNaN(v)) return 0;
    if (v === Infinity) return 34028235e31;
    if (v === -Infinity) return -34028235e31;
    return v;
  },
  (a, aFn, dz) => {
    aFn.backward(dz);
  },
  "nan_to_num"
);
class Reshape extends TorchFunction {
  _forward(a, shape) {
    const previous_length = a.dataLength();
    const target_length = shape.reduce((acc, val) => acc * val, 1);
    if (previous_length !== target_length) {
      throw new Error("Shape mismatch: " + a.shape + " and " + shape);
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
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.reshape(a.shape));
  }
}
registerOperation("reshape", Reshape);
class Squeeze extends TorchFunction {
  _forward(a, dim) {
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
    if (dim !== void 0) {
      if (dim < 0) {
        dim += a.shape.length;
      }
      if (shape[dim] === 1) {
        shape.splice(dim, 1);
      }
    } else {
      shape = shape.filter((d) => d !== 1);
    }
    return new Tensor(
      a.data,
      { requires_grad: rg },
      { operation: rg ? this : null, shape }
    );
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.reshape(a.shape));
  }
}
registerOperation("squeeze", Squeeze);
class Unsqueeze extends TorchFunction {
  _forward(a, dim) {
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
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(dz.reshape(a.shape));
  }
}
registerOperation("unsqueeze", Unsqueeze);
class Expand extends TorchFunction {
  _forward(a, expanded_shape) {
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
    const outData = broadcast(a, target_shape).data;
    return new Tensor(
      outData,
      { requires_grad: rg },
      { operation: rg ? this : null, shape: target_shape }
    );
  }
  _backward(dz) {
    const [a] = this.saved_tensors;
    const [aFn] = this.next_functions;
    aFn.backward(unbroadcast(dz, a.shape));
  }
}
registerOperation("expand", Expand);
UnaryFunctionMixin(
  (a, x) => Math.sin(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.cos()));
  },
  "sin"
);
UnaryFunctionMixin(
  (a, x) => Math.cos(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.sin().neg()));
  },
  "cos"
);
UnaryFunctionMixin(
  (a, x) => Math.tan(a[x]),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.cos().pow(-2)));
  },
  "tan"
);
const Sum = ReductionFunctionMixin(
  0,
  (acc, val) => acc + val,
  (a, expanded_dz) => expanded_dz,
  "sum"
);
const Mean = ReductionFunctionMixin(
  0,
  (acc, val) => acc + val,
  (a, expanded_dz, dim) => {
    const target_shape = _get_reduction_shape(a.shape, dim, false);
    const out_size = target_shape.length > 0 ? target_shape.reduce((acc, v) => acc * v, 1) : 1;
    const N = a.dataLength() / out_size;
    return expanded_dz.mul(new Tensor([1 / N]));
  },
  "mean",
  (acc, count) => acc / count
);
const Max = ReductionFunctionMixin(
  -Infinity,
  (acc, val) => Math.max(acc, val),
  (a, expanded_dz, dim) => {
    const max_tensor = a.max(dim, true);
    const max_expanded = max_tensor.expand(a.shape);
    const mask = a.eq(max_expanded).detach();
    return expanded_dz.mul(mask);
  },
  "max"
);
const Min = ReductionFunctionMixin(
  Infinity,
  (acc, val) => Math.min(acc, val),
  (a, expanded_dz, dim) => {
    const min_tensor = a.min(dim, true);
    const min_expanded = min_tensor.expand(a.shape);
    const mask = a.eq(min_expanded).detach();
    return expanded_dz.mul(mask);
  },
  "min"
);
function _transpose_tensor(a, dim0, dim1, operation = null) {
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
    { requires_grad: resultRequiresGrad(a) },
    { operation, shape: output_shape }
  );
}
class Transpose extends TorchFunction {
  dim0;
  dim1;
  _forward(a, dim0, dim1) {
    const rg = resultRequiresGrad(a);
    if (rg) {
      this.saved_tensors = [a];
      this.dim0 = dim0;
      this.dim1 = dim1;
    }
    this.next_functions.push(a.grad_fn ? a.grad_fn : nullOp);
    return _transpose_tensor(a, dim0, dim1, rg ? this : null);
  }
  _backward(dz) {
    const dim0 = this.dim0;
    const dim1 = this.dim1;
    const [aFn] = this.next_functions;
    aFn.backward(dz.transpose(dim0, dim1));
  }
}
registerOperation("transpose", Transpose);
function _matmul_tensor(a, b, operation = null) {
  if (a.shape.length == 1 && b.shape.length == 1) {
    return [a.mul(b).sum(), []];
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
    const base_a = _get_original_index(padded_a_shape, broadcast_shape, i - n);
    const base_b = _get_original_index(padded_b_shape, broadcast_shape, i - m * dim_N);
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
  return [new Tensor(
    data,
    { requires_grad: resultRequiresGrad(a, b) },
    { operation, shape: shape_after_removing_extra_dims }
  ), shape_after_removing_extra_dims];
}
class Matmul extends BinaryFunction {
  shape;
  _forward(a, b) {
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
  _backward(dz) {
    const [a, b] = this.saved_tensors;
    const [aFn, bFn] = this.next_functions;
    if (a.shape.length === 1 && b.shape.length === 1) {
      aFn.backward(dz.mul(b));
      bFn.backward(dz.mul(a));
      return;
    }
    if (a.shape.length === 1) {
      const dz1 = dz.unsqueeze(-2);
      const a1 = a.unsqueeze(-2);
      let da2 = dz1.matmul(b.transpose(-2, -1));
      let db2 = a1.transpose(-2, -1).matmul(dz1);
      da2 = da2.squeeze(-2);
      db2 = unbroadcast(db2, b.shape);
      aFn.backward(da2);
      bFn.backward(db2);
      return;
    }
    if (b.shape.length === 1) {
      const dz1 = dz.unsqueeze(-1);
      const b1 = b.unsqueeze(-1);
      let da2 = dz1.matmul(b1.transpose(-2, -1));
      let db2 = a.transpose(-2, -1).matmul(dz1);
      da2 = unbroadcast(da2, a.shape);
      db2 = db2.squeeze(-1);
      aFn.backward(da2);
      bFn.backward(db2);
      return;
    }
    let da = dz.matmul(b.transpose(-2, -1));
    let db = a.transpose(-2, -1).matmul(dz);
    da = unbroadcast(da, a.shape);
    db = unbroadcast(db, b.shape);
    aFn.backward(da);
    bFn.backward(db);
  }
}
registerOperation("matmul", Matmul);
function _convNd_forward(input, weight, bias, stride, padding, dilation, groups, dims) {
  const stride_arr = typeof stride === "number" ? new Array(dims).fill(stride) : stride;
  const padding_arr = typeof padding === "number" ? new Array(dims).fill(padding) : padding;
  const dilation_arr = typeof dilation === "number" ? new Array(dims).fill(dilation) : dilation;
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
  const get_strides = (shape) => {
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
  for (let b = 0; b < batch_size; b++) {
    for (let g = 0; g < groups; g++) {
      for (let oc_g = 0; oc_g < out_channels_per_group; oc_g++) {
        const oc = g * out_channels_per_group + oc_g;
        const out_spatial_size = out_dims.reduce((a, b2) => a * b2, 1);
        for (let os_idx = 0; os_idx < out_spatial_size; os_idx++) {
          const os_coords = new Array(dims);
          let temp_os = os_idx;
          for (let d = dims - 1; d >= 0; d--) {
            os_coords[d] = temp_os % out_dims[d];
            temp_os = Math.floor(temp_os / out_dims[d]);
          }
          let sum2 = bias ? bias.data[oc] : 0;
          for (let ic_g = 0; ic_g < in_channels_per_group; ic_g++) {
            const ic = g * in_channels_per_group + ic_g;
            const kernel_spatial_size = kernel_dims.reduce((a, b2) => a * b2, 1);
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
                sum2 += input.data[in_flat_idx] * weight.data[w_flat_idx];
              }
            }
          }
          let out_flat_idx = b * out_strides[0] + oc * out_strides[1];
          for (let d = 0; d < dims; d++) out_flat_idx += os_coords[d] * out_strides[d + 2];
          output_data[out_flat_idx] = sum2;
        }
      }
    }
  }
  return new Tensor(output_data, { requires_grad: false }, { shape: output_shape });
}
function _convNd_backward(dz, input, weight, bias, stride, padding, dilation, groups, dims, input_requires_grad, weight_requires_grad) {
  const stride_arr = typeof stride === "number" ? new Array(dims).fill(stride) : stride;
  const padding_arr = typeof padding === "number" ? new Array(dims).fill(padding) : padding;
  const dilation_arr = typeof dilation === "number" ? new Array(dims).fill(dilation) : dilation;
  const batch_size = input.shape[0];
  const in_channels = input.shape[1];
  const out_channels = weight.shape[0];
  const in_dims = input.shape.slice(2);
  const kernel_dims = weight.shape.slice(2);
  const out_dims = dz.shape.slice(2);
  const get_strides = (shape) => {
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
  let dInput = null;
  let dWeight = null;
  let dBias = null;
  let dInput_data = null;
  let dWeight_data = null;
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
        const out_spatial_size = out_dims.reduce((a, b2) => a * b2, 1);
        for (let os_idx = 0; os_idx < out_spatial_size; os_idx++) {
          const os_coords = new Array(dims);
          let temp_os = os_idx;
          for (let d = dims - 1; d >= 0; d--) {
            os_coords[d] = temp_os % out_dims[d];
            temp_os = Math.floor(temp_os / out_dims[d]);
          }
          let dz_flat_idx = b * dz_strides[0] + oc * dz_strides[1];
          for (let d = 0; d < dims; d++) dz_flat_idx += os_coords[d] * dz_strides[d + 2];
          const dz_val = dz.data[dz_flat_idx];
          for (let ic_g = 0; ic_g < in_channels_per_group; ic_g++) {
            const ic = g * in_channels_per_group + ic_g;
            const kernel_spatial_size = kernel_dims.reduce((a, b2) => a * b2, 1);
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
                  dInput_data[in_flat_idx] += dz_val * weight.data[w_flat_idx];
                }
                if (weight_requires_grad) {
                  dWeight_data[w_flat_idx] += dz_val * input.data[in_flat_idx];
                }
              }
            }
          }
        }
      }
    }
  }
  if (input_requires_grad) dInput = new Tensor(dInput_data, { requires_grad: false }, { shape: input.shape });
  if (weight_requires_grad) dWeight = new Tensor(dWeight_data, { requires_grad: false }, { shape: weight.shape });
  if (bias && bias.requires_grad) {
    const sum_dims = [0];
    for (let d = 2; d < dz.shape.length; d++) sum_dims.push(d);
    dBias = dz.sum(sum_dims);
  }
  return [dInput, dWeight, dBias];
}
class Conv1dOp extends TorchFunction {
  stride;
  padding;
  dilation;
  groups;
  _forward(input, weight, bias, stride = 1, padding = 0, dilation = 1, groups = 1) {
    const rg = resultRequiresGrad(input, weight, ...bias ? [bias] : []);
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
  _backward(dz) {
    const input = this.saved_tensors[0];
    const weight = this.saved_tensors[1];
    const bias = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null;
    const [inputFn, weightFn, biasFn] = this.next_functions;
    const [dInput, dWeight, dBias] = _convNd_backward(
      dz,
      input,
      weight,
      bias,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      1,
      input.requires_grad,
      weight.requires_grad
    );
    if (input.requires_grad) inputFn.backward(dInput);
    if (weight.requires_grad) weightFn.backward(dWeight);
    if (bias && bias.requires_grad) biasFn.backward(dBias);
  }
}
registerOperation("conv1d", Conv1dOp);
class Conv2dOp extends TorchFunction {
  stride;
  padding;
  dilation;
  groups;
  _forward(input, weight, bias, stride = 1, padding = 0, dilation = 1, groups = 1) {
    const rg = resultRequiresGrad(input, weight, ...bias ? [bias] : []);
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
  _backward(dz) {
    const input = this.saved_tensors[0];
    const weight = this.saved_tensors[1];
    const bias = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null;
    const [inputFn, weightFn, biasFn] = this.next_functions;
    const [dInput, dWeight, dBias] = _convNd_backward(
      dz,
      input,
      weight,
      bias,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      2,
      input.requires_grad,
      weight.requires_grad
    );
    if (input.requires_grad) inputFn.backward(dInput);
    if (weight.requires_grad) weightFn.backward(dWeight);
    if (bias && bias.requires_grad) biasFn.backward(dBias);
  }
}
registerOperation("conv2d", Conv2dOp);
class Conv3dOp extends TorchFunction {
  stride;
  padding;
  dilation;
  groups;
  _forward(input, weight, bias, stride = 1, padding = 0, dilation = 1, groups = 1) {
    const rg = resultRequiresGrad(input, weight, ...bias ? [bias] : []);
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
  _backward(dz) {
    const input = this.saved_tensors[0];
    const weight = this.saved_tensors[1];
    const bias = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null;
    const [inputFn, weightFn, biasFn] = this.next_functions;
    const [dInput, dWeight, dBias] = _convNd_backward(
      dz,
      input,
      weight,
      bias,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      3,
      input.requires_grad,
      weight.requires_grad
    );
    if (input.requires_grad) inputFn.backward(dInput);
    if (weight.requires_grad) weightFn.backward(dWeight);
    if (bias && bias.requires_grad) biasFn.backward(dBias);
  }
}
registerOperation("conv3d", Conv3dOp);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] < b[b_index] ? 1 : 0,
  () => {
  },
  "lt"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] > b[b_index] ? 1 : 0,
  () => {
  },
  "gt"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] <= b[b_index] ? 1 : 0,
  () => {
  },
  "le"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] >= b[b_index] ? 1 : 0,
  () => {
  },
  "ge"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] == b[b_index] ? 1 : 0,
  () => {
  },
  "eq"
);
BinaryFunctionMixin(
  (a, b, a_index, b_index) => a[a_index] != b[b_index] ? 1 : 0,
  () => {
  },
  "ne"
);
UnaryFunctionMixin(
  (a, x) => Math.max(a[x], 0),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.gt(0)));
  },
  "relu"
);
UnaryFunctionMixin(
  (a, x) => 1 / (1 + Math.exp(-a[x])),
  (a, aFn, dz) => {
    const res = a.sigmoid();
    aFn.backward(res.mul(res.mul(-1).add(1)).mul(dz));
  },
  "sigmoid"
);
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
  named_parameters(prefix = "") {
    const result = [];
    for (const [name, param] of Object.entries(this._parameters)) {
      const fullName = prefix ? `${prefix}.${name}` : name;
      result.push([fullName, param]);
    }
    for (const [name, module] of Object.entries(this._modules)) {
      const fullName = prefix ? `${prefix}.${name}` : name;
      result.push(...module.named_parameters(fullName));
    }
    return result;
  }
}
class Sequential extends Module {
  _modulesArr;
  constructor(...modules) {
    super();
    this._modulesArr = modules;
    for (let i = 0; i < modules.length; i++) {
      this.register(i.toString(), modules[i]);
    }
  }
  append(module) {
    this.register(this._modulesArr.length.toString(), module);
    this._modulesArr.push(module);
    return this;
  }
  extend(sequential) {
    for (const module of sequential._modulesArr) {
      this.append(module);
    }
    return this;
  }
  insert(index2, module) {
    this._modulesArr.splice(index2, 0, module);
    for (let i = index2; i < this._modulesArr.length; i++) {
      this.register(i.toString(), this._modulesArr[i]);
    }
    return this;
  }
  forward(input) {
    let x = input;
    for (const module of this._modulesArr) {
      x = module.forward(x);
    }
    return x;
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
function generate_function(opname) {
  return (...args) => {
    const operation = createOperation(opname);
    return operation.forward(...args);
  };
}
function generate_unary_function(opname) {
  return (a) => {
    if (typeof a == "number") {
      a = new Tensor(a);
    }
    const operation = createOperation(opname);
    return operation.forward(a);
  };
}
const relu = generate_unary_function("relu");
const sigmoid = generate_unary_function("sigmoid");
const conv1d = generate_function("conv1d");
const conv2d = generate_function("conv2d");
const conv3d = generate_function("conv3d");
const functional = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  conv1d,
  conv2d,
  conv3d,
  relu,
  sigmoid
}, Symbol.toStringTag, { value: "Module" }));
class Linear extends Module {
  weight;
  bias;
  constructor(in_features, out_features) {
    super();
    const k = Math.sqrt(1 / in_features);
    this.weight = new Parameter(
      rand([out_features, in_features]).mul(2 * k).sub(k)
    );
    this.bias = new Parameter(
      rand([out_features]).mul(2 * k).sub(k)
    );
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
class Sigmoid extends Module {
  constructor() {
    super();
  }
  forward(input) {
    return sigmoid(input);
  }
}
class _ConvNd extends Module {
  weight;
  bias;
  in_channels;
  out_channels;
  kernel_size;
  stride;
  padding;
  dilation;
  groups;
  constructor(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dims) {
    super();
    this.in_channels = in_channels;
    this.out_channels = out_channels;
    this.kernel_size = kernel_size;
    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;
    if (in_channels % groups !== 0) {
      throw new Error("in_channels must be divisible by groups");
    }
    if (out_channels % groups !== 0) {
      throw new Error("out_channels must be divisible by groups");
    }
    const kernel_arr = typeof kernel_size === "number" ? new Array(dims).fill(kernel_size) : kernel_size;
    const kernel_vol = kernel_arr.reduce((a, b) => a * b, 1);
    const k = Math.sqrt(groups / (in_channels * kernel_vol));
    this.weight = new Parameter(
      rand([out_channels, in_channels / groups, ...kernel_arr]).mul(2 * k).sub(k)
    );
    this.register("weight", this.weight);
    if (bias) {
      this.bias = new Parameter(
        rand([out_channels]).mul(2 * k).sub(k)
      );
      this.register("bias", this.bias);
    } else {
      this.bias = null;
    }
  }
}
class Conv1d extends _ConvNd {
  constructor(in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = true) {
    super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 1);
  }
  forward(input) {
    return conv1d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
class Conv2d extends _ConvNd {
  constructor(in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = true) {
    super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 2);
  }
  forward(input) {
    return conv2d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
class Conv3d extends _ConvNd {
  constructor(in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = true) {
    super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 3);
  }
  forward(input) {
    return conv3d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
const index$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss,
  Conv1d,
  Conv2d,
  Conv3d,
  L1Loss,
  Linear,
  MSELoss,
  Module,
  Parameter,
  ReLU,
  Sequential,
  Sigmoid,
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
        const buf = this.state.get(param).velocity;
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
const _atenMap = {
  "add": "aten.add.Tensor",
  "sub": "aten.sub.Tensor",
  "mul": "aten.mul.Tensor",
  "div": "aten.div.Tensor",
  "pow": "aten.pow.Tensor_Tensor",
  "powint": "aten.pow.Tensor_Scalar",
  "fmod": "aten.fmod.Tensor",
  "maximum": "aten.maximum.default",
  "minimum": "aten.minimum.default",
  "log": "aten.log.default",
  "sqrt": "aten.sqrt.default",
  "exp": "aten.exp.default",
  "square": "aten.square.default",
  "abs": "aten.abs.default",
  "sign": "aten.sign.default",
  "neg": "aten.neg.default",
  "reciprocal": "aten.reciprocal.default",
  "nan_to_num": "aten.nan_to_num.default",
  "reshape": "aten.reshape.default",
  "squeeze": "aten.squeeze.dim",
  "unsqueeze": "aten.unsqueeze.default",
  "expand": "aten.expand.default",
  "sin": "aten.sin.default",
  "cos": "aten.cos.default",
  "tan": "aten.tan.default",
  "sum": "aten.sum.default",
  "mean": "aten.mean.default",
  "min": "aten.min.default",
  "max": "aten.max.default",
  "transpose": "aten.transpose.int",
  "matmul": "aten.matmul.default",
  "relu": "aten.relu.default",
  "sigmoid": "aten.sigmoid.default",
  "lt": "aten.lt.Tensor",
  "gt": "aten.gt.Tensor",
  "le": "aten.le.Tensor",
  "ge": "aten.ge.Tensor",
  "eq": "aten.eq.Tensor",
  "ne": "aten.ne.Tensor",
  "conv1d": "aten.conv1d.default",
  "conv2d": "aten.conv2d.default",
  "conv3d": "aten.conv3d.default",
  "linear": "aten.linear.default"
};
function toAtenTarget(opName) {
  return _atenMap[opName] || `aten.${opName}.default`;
}
class NameGenerator {
  counts = /* @__PURE__ */ new Map();
  generate(baseName) {
    const count = this.counts.get(baseName) || 0;
    this.counts.set(baseName, count + 1);
    return count === 0 ? baseName : `${baseName}_${count}`;
  }
}
class ExportedProgram {
  constructor(graph, graph_signature, parameters) {
    this.graph = graph;
    this.graph_signature = graph_signature;
    this.parameters = parameters;
  }
  toString() {
    const lines = ["ExportedProgram:"];
    const inputArgs = this.graph.filter((n) => n.op === "placeholder").map((n) => {
      const shape = n.val_shape ? JSON.stringify(n.val_shape) : "?";
      return `${n.name}: "${shape}"`;
    }).join(", ");
    lines.push(`    class GraphModule(torch.nn.Module):`);
    lines.push(`        def forward(self, ${inputArgs}):`);
    for (const node of this.graph) {
      if (node.op === "call_function") {
        const args = node.args.join(", ");
        lines.push(`            ${node.name} = ${node.target}(${args})`);
      } else if (node.op === "output") {
        lines.push(`            return (${node.args.join(", ")},)`);
      }
    }
    lines.push("");
    lines.push("Graph signature:");
    lines.push("    # inputs");
    for (const spec of this.graph_signature.input_specs) {
      const target = spec.target ? ` target='${spec.target}'` : "";
      lines.push(`    ${spec.name}: ${spec.kind}${target}`);
    }
    lines.push("    # outputs");
    for (const spec of this.graph_signature.output_specs) {
      lines.push(`    ${spec.name}: ${spec.kind}`);
    }
    return lines.join("\n");
  }
}
function export_(module, sampleInputs) {
  const graph = [];
  const nameGen = new NameGenerator();
  const tensorIdToName = /* @__PURE__ */ new Map();
  const namedParams = module.named_parameters();
  const paramTensorIds = /* @__PURE__ */ new Set();
  const inputSpecs = [];
  for (const [paramPath, param] of namedParams) {
    const placeholderName = "p_" + paramPath.replace(/\./g, "_");
    const nodeName = nameGen.generate(placeholderName);
    tensorIdToName.set(param.id, nodeName);
    paramTensorIds.add(param.id);
    graph.push({
      op: "placeholder",
      name: nodeName,
      target: nodeName,
      args: [],
      val_shape: param.shape
    });
    inputSpecs.push({
      kind: "PARAMETER",
      name: nodeName,
      target: paramPath
    });
  }
  for (let i = 0; i < sampleInputs.length; i++) {
    const baseName = "input";
    const nodeName = nameGen.generate(baseName);
    tensorIdToName.set(sampleInputs[i].id, nodeName);
    graph.push({
      op: "placeholder",
      name: nodeName,
      target: nodeName,
      args: [],
      val_shape: sampleInputs[i].shape
    });
    inputSpecs.push({
      kind: "USER_INPUT",
      name: nodeName
    });
  }
  const handler = (e) => {
    const { operation, args, result } = e.detail;
    const opName = operation.opName;
    if (!opName) return;
    const nodeArgs = [];
    for (const arg of args) {
      if (arg instanceof Tensor) {
        const name = tensorIdToName.get(arg.id);
        if (name) {
          nodeArgs.push(name);
        }
      }
    }
    const nodeName = nameGen.generate(opName);
    tensorIdToName.set(result.id, nodeName);
    graph.push({
      op: "call_function",
      name: nodeName,
      target: toAtenTarget(opName),
      args: nodeArgs,
      val_shape: result.shape
    });
  };
  eventBus.addEventListener(
    events.OPERATION_AFTER_FORWARD,
    handler
  );
  let output;
  try {
    output = no_grad(() => module.forward(...sampleInputs));
  } finally {
    eventBus.removeEventListener(
      events.OPERATION_AFTER_FORWARD,
      handler
    );
  }
  const outputName = tensorIdToName.get(output.id) || "output";
  graph.push({
    op: "output",
    name: "output",
    target: "output",
    args: [outputName]
  });
  const outputSpecs = [{
    kind: "USER_OUTPUT",
    name: outputName
  }];
  const parameters = /* @__PURE__ */ new Map();
  for (const [paramPath, param] of namedParams) {
    parameters.set(paramPath, {
      data: [...param.data],
      shape: [...param.shape]
    });
  }
  return new ExportedProgram(
    graph,
    { input_specs: inputSpecs, output_specs: outputSpecs },
    parameters
  );
}
export {
  AccumulateGrad,
  ExportedProgram,
  Max,
  Mean,
  Min,
  Sum,
  Tensor,
  TorchFunction,
  __left_index__,
  __right_index__,
  abs,
  add,
  allclose,
  arange,
  cos,
  disable_no_grad,
  div,
  enable_no_grad,
  eq,
  eventBus,
  events,
  exp,
  expand,
  export_,
  fmod,
  ge,
  gt,
  is_grad_enabled,
  le,
  linspace,
  log,
  lt,
  manual_seed,
  matmul,
  max,
  maximum,
  mean,
  min,
  minimum,
  mul,
  nan_to_num,
  ne,
  neg,
  index$1 as nn,
  no_grad,
  numel,
  ones,
  ones_like,
  index as optim,
  pow,
  rand,
  randint,
  randn,
  randperm,
  reciprocal,
  reshape,
  seed,
  sign,
  sin,
  sqrt,
  square,
  squeeze,
  sub,
  sum,
  tan,
  tensor,
  transpose,
  unsqueeze,
  zeros,
  zeros_like
};
//# sourceMappingURL=torch.node.es.mjs.map
