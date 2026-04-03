import { AccumulateGrad, TorchFunction, resultRequiresGrad } from './functions/base';
import { getOperationCache, createOperation } from './functions/registry';
import { getNextId, eventBus, events } from './util';

export type TypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export type NestedNumberArray = number | TypedArray | NestedNumberArray[];

function _get_shape(data: NestedNumberArray): number[] {
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

function _assert_shape(data: NestedNumberArray, shape: number[]): void {
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

function _get_and_assert_shape(data: NestedNumberArray): number[] {
  const shape = _get_shape(data);
  _assert_shape(data, shape);
  return shape;
}

function _flatten(data: NestedNumberArray): number[] {
  if (Array.isArray(data)) {
    return data.flatMap(item => _flatten(item));
  } else if (ArrayBuffer.isView(data)) {
    return Array.from(data);
  } else {
    return [data];
  }
}

/**
 * A shared backing store for tensor data.
 * Multiple tensors (views) may reference the same TensorStorage instance.
 * Mutating `data` on the TensorStorage is visible to all sharing tensors.
 */
export class TensorStorage {
  constructor(public data: number[]) {}
}

export class Tensor {
  // Auto-generated ID
  public id: number = getNextId();

  // Optional user-defined name
  public name: string | null = null;

  // Shared backing storage and offset into it.
  // Views share the same TensorStorage but differ in _offset and shape.
  private _storage: TensorStorage = new TensorStorage([]);
  private _offset: number = 0;

  /**
   * Returns the flat, contiguous data for this tensor.
   *
   * Fast path (non-view): returns the storage array directly — no allocation.
   * View path: materialises a contiguous slice — one allocation per call,
   * so callers inside tight loops should cache the result: `const d = t.data`.
   */
  get data(): number[] {
    const n = this.dataLength();
    if (this._offset === 0 && this._storage.data.length === n) {
      return this._storage.data;
    }
    return this._storage.data.slice(this._offset, this._offset + n);
  }

  /**
   * Sets the tensor's data.
   *
   * Non-view (offset=0, storage covers exactly this tensor's numel):
   *   replaces the shared storage's data array in-place — all other views
   *   sharing the same TensorStorage immediately see the new values.
   *
   * View (offset≠0 or storage is larger than this tensor):
   *   copies `values` element-by-element into the shared storage at the
   *   correct offset — the original tensor and sibling views are updated.
   */
  set data(values: number[]) {
    const n = values.length;
    if (this._offset === 0 && this._storage.data.length === n) {
      // Full-storage owner: swap out the backing array.
      this._storage.data = values;
    } else {
      // View: write into shared storage at the right offset.
      for (let i = 0; i < n; i++) {
        this._storage.data[this._offset + i] = values[i];
      }
    }
  }

  public shape: number[];
  public grad_fn: TorchFunction | null = null;
  public grad: Tensor | null = null;

  public requires_grad: boolean;

  constructor(
    data: NestedNumberArray,
    options: { requires_grad?: boolean; name?: string } = {},
    internal_options: {
      operation?: TorchFunction;
      shape?: number[];
      /** For internal view construction only — share an existing storage. */
      _storage?: TensorStorage;
      /** Byte offset into _storage (in elements). */
      _offset?: number;
    } = {}
  ) {
    if (internal_options._storage !== undefined) {
      // View construction: share the provided storage.
      this._storage = internal_options._storage;
      this._offset = internal_options._offset ?? 0;
      this.shape = internal_options.shape ?? [];
    } else {
      this._storage = new TensorStorage(_flatten(data));
      this._offset = 0;
      this.shape = internal_options.shape ?? _get_and_assert_shape(data);
    }

    this.requires_grad = options.requires_grad ?? false;

    if (options.name) {
      this.name = options.name;
    }

    this.grad_fn = internal_options.operation ?? null;

    if (this.requires_grad && !this.grad_fn) {
      const acc = new AccumulateGrad();
      acc.variable = this;
      this.grad_fn = acc;
    }
  }

  size(dim?: number): number | number[] {
    if (dim !== undefined) {
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

  toArray_(): void {
    return;
  }

  toFlatArray(): number[] {
    return this.data;
  }

  toArray(): NestedNumberArray {
    if (this.shape.length === 0) {
      return this.data[0];
    }

    let flatIndex = 0;
    const flatData = this.data;

    const buildDimension = (currentDim: number): NestedNumberArray => {
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

  toString(): string {
    let extra = '';
    if (this.name) {
      extra += `, name="${this.name}"`;
    }
    if (this.dataLength() == 0 && this.shape.length > 0) {
      extra += `, size=(${this.shape.join(', ')})`;
    }
    if (this.requires_grad) {
      extra += ', requires_grad=True';
    }

    function formatNum(val: number): string {
      return String(Math.round(val * 1e4) / 1e4);
    }

    function formatArray(val: unknown): string {
      if (Array.isArray(val)) {
        return "[" + val.map(formatArray).join(", ") + "]";
      }
      if (typeof val === "number") {
        return formatNum(val);
      }
      return String(val);
    }

    return `tensor(${formatArray(this.toArray())}${extra})`;
  }

  dataLength(): number {
    if (this.shape.length === 0) return 1;
    return this.shape.reduce((a, b) => a * b, 1);
  }

  private _executeUnaryOp(opName: string): Tensor {
    const operation = resultRequiresGrad(this)
      ? createOperation(opName)
      : getOperationCache(opName);
    return operation.forward(this);
  }

  private _executeBinaryOp(opName: string, other: Tensor | number): Tensor {
    if (typeof other == 'number') {
      other = new Tensor(other);
    }
    const operation = resultRequiresGrad(this, other)
      ? createOperation(opName)
      : getOperationCache(opName);
    return operation.forward(this, other);
  }

  private _executeOpRaw(opName: string, ...args: any[]): Tensor {
    const operation = createOperation(opName);
    return operation.forward(this, ...args);
  }

  item(): number {
    if (this.dataLength() !== 1) {
      throw new Error('Tensor.item() is only valid for scalars');
    }
    return this.data[0];
  }

  detach(): Tensor {
    return new Tensor(this.data, { requires_grad: false }, { shape: this.shape });
  }

  detach_(): void {
    this.requires_grad = false;
    this.grad = null;
    this.grad_fn = null;
  }

  zero_(): void {
    this.data = Array(this.dataLength()).fill(0);
  }

  private is_retain_grad: boolean = false;
  retain_grad(): void {
    // leaf node -> no-op
    if (this.grad_fn instanceof AccumulateGrad) return;
    if (this.is_retain_grad) return;
    this.is_retain_grad = true;

    this.grad_fn._retained_tensors.push(this);
  }

  backward(grad?: Tensor | null): void {
    if (!this.requires_grad) {
      return;
    }

    if (!grad) {
      if (this.dataLength() !== 1) {
        throw new Error('Gradient is required for non-scalar tensors');
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

  /**
   * Returns a view of this tensor along dimension 0.
   *
   * The returned tensor shares the same underlying TensorStorage — mutations
   * to either tensor (via zero_(), the data setter, or the optimizer) are
   * immediately visible in the other.
   *
   * Supports negative indices (e.g. index(-1) is the last row).
   *
   * Note: the view does not carry a grad_fn; autograd does not propagate
   * through index() at this time.
   */
  index(i: number): Tensor {
    if (this.shape.length === 0) {
      throw new Error('Cannot index a scalar tensor');
    }
    if (i < 0) {
      i += this.shape[0];
    }
    if (i < 0 || i >= this.shape[0]) {
      throw new Error(
        `Index ${i} out of bounds for dimension 0 with size ${this.shape[0]}`
      );
    }
    const newShape = this.shape.slice(1);
    // Number of elements per row along dim 0.
    const rowSize = newShape.length === 0 ? 1 : newShape.reduce((a, b) => a * b, 1);
    const newOffset = this._offset + i * rowSize;
    return new Tensor([], {}, { shape: newShape, _storage: this._storage, _offset: newOffset });
  }

  // operations

  // binary pointwise

  add(other: Tensor | number): Tensor {
    return this._executeBinaryOp('add', other);
  }

  sub(other: Tensor | number): Tensor {
    return this._executeBinaryOp('sub', other);
  }

  mul(other: Tensor | number): Tensor {
    return this._executeBinaryOp('mul', other);
  }

  div(other: Tensor | number): Tensor {
    return this._executeBinaryOp('div', other);
  }

  pow(other: Tensor | number): Tensor {
    if (typeof other == 'number' && other % 1 === 0) {
      return this._executeOpRaw('powint', other);
    }
    return this._executeBinaryOp('pow', other);
  }

  fmod(other: Tensor | number): Tensor {
    return this._executeBinaryOp('fmod', other);
  }

  maximum(other: Tensor | number): Tensor {
    return this._executeBinaryOp('maximum', other);
  }

  minimum(other: Tensor | number): Tensor {
    return this._executeBinaryOp('minimum', other);
  }

  // unary pointwise

  log(): Tensor {
    return this._executeUnaryOp('log');
  }

  sqrt(): Tensor {
    return this._executeUnaryOp('sqrt');
  }

  exp(): Tensor {
    return this._executeUnaryOp('exp');
  }

  square(): Tensor {
    return this._executeUnaryOp('square');
  }

  abs(): Tensor {
    return this._executeUnaryOp('abs');
  }

  sign(): Tensor {
    return this._executeUnaryOp('sign');
  }

  neg(): Tensor {
    return this._executeUnaryOp('neg');
  }

  reciprocal(): Tensor {
    return this._executeUnaryOp('reciprocal');
  }

  nan_to_num(): Tensor {
    return this._executeUnaryOp('nan_to_num');
  }

  reshape(shape: number[]): Tensor {
    return this._executeOpRaw('reshape', shape);
  }

  flatten(start_dim: number = 0, end_dim: number = -1): Tensor {
    return this._executeOpRaw('flatten', start_dim, end_dim);
  }

  squeeze(dim: number): Tensor {
    return this._executeOpRaw('squeeze', dim);
  }

  unsqueeze(dim: number): Tensor {
    return this._executeOpRaw('unsqueeze', dim);
  }

  expand(sizes: number[]): Tensor {
    return this._executeOpRaw('expand', sizes);
  }

  // trigonometric

  sin(): Tensor {
    return this._executeUnaryOp('sin');
  }

  cos(): Tensor {
    return this._executeUnaryOp('cos');
  }

  tan(): Tensor {
    return this._executeUnaryOp('tan');
  }

  // reduction

  sum(dim?: number | number[], keepdim: boolean = false): Tensor {
    return this._executeOpRaw('sum', dim, keepdim);
  }

  mean(dim?: number | number[], keepdim: boolean = false): Tensor {
    return this._executeOpRaw('mean', dim, keepdim);
  }

  max(dim?: number | number[], keepdim: boolean = false): Tensor {
    return this._executeOpRaw('max', dim, keepdim);
  }

  min(dim?: number | number[], keepdim: boolean = false): Tensor {
    return this._executeOpRaw('min', dim, keepdim);
  }

  // linalg

  transpose(dim0: number, dim1: number): Tensor {
    return this._executeOpRaw('transpose', dim0, dim1);
  }

  matmul(other: Tensor): Tensor {
    return this._executeBinaryOp('matmul', other);
  }

  // comparison

  lt(other: Tensor | number): Tensor {
    return this._executeBinaryOp('lt', other);
  }

  gt(other: Tensor | number): Tensor {
    return this._executeBinaryOp('gt', other);
  }

  le(other: Tensor | number): Tensor {
    return this._executeBinaryOp('le', other);
  }

  ge(other: Tensor | number): Tensor {
    return this._executeBinaryOp('ge', other);
  }

  eq(other: Tensor | number): Tensor {
    return this._executeBinaryOp('eq', other);
  }

  ne(other: Tensor | number): Tensor {
    return this._executeBinaryOp('ne', other);
  }

  allclose(
    other: Tensor,
    rtol: number = 1e-5,
    atol: number = 1e-8,
    equal_nan: boolean = false
  ): boolean {
    const thisData = this.data;
    const otherData = other.data;
    if (thisData.length !== otherData.length) return false;
    for (let i = 0; i < thisData.length; i++) {
      const av = thisData[i],
        bv = otherData[i];
      if (equal_nan && isNaN(av) && isNaN(bv)) continue;
      if (isNaN(av) || isNaN(bv)) return false;
      if (Math.abs(av - bv) > atol + rtol * Math.abs(bv)) return false;
    }
    return true;
  }

  numel(): number {
    return this.dataLength();
  }

  // other

  sigmoid(): Tensor {
    return this._executeUnaryOp('sigmoid');
  }

  relu(): Tensor {
    return this._executeUnaryOp('relu');
  }

  softmax(dim: number): Tensor {
    return this._executeOpRaw('softmax', dim);
  }

  clamp(min: number, max: number): Tensor {
    return this._executeOpRaw('clamp', min, max);
  }

  cat(tensors: Tensor | Tensor[], dim: number = 0): Tensor {
    const others = Array.isArray(tensors) ? tensors : [tensors];
    return createOperation('cat').forward([this, ...others], dim);
  }

  concatenate(tensors: Tensor | Tensor[], dim: number = 0): Tensor {
    return this.cat(tensors, dim);
  }

  concat(tensors: Tensor | Tensor[], dim: number = 0): Tensor {
    return this.cat(tensors, dim);
  }
}

// ---------------------------------------------------------------------------
// Typed tensor constructors
// ---------------------------------------------------------------------------

function _truncate_nested(data: NestedNumberArray): NestedNumberArray {
  if (typeof data === 'number') return Math.trunc(data);
  if (Array.isArray(data)) return (data as NestedNumberArray[]).map(_truncate_nested);
  // TypedArray
  const out = new Float64Array((data as Float64Array).length);
  for (let i = 0; i < out.length; i++) out[i] = Math.trunc((data as Float64Array)[i]);
  return out;
}

/**
 * A Tensor that stores 32-bit float values (same as the default Tensor).
 * Provided for PyTorch API compatibility.
 */
export class FloatTensor extends Tensor {
  constructor(data: NestedNumberArray, options: { requires_grad?: boolean } = {}) {
    super(data, options);
  }
}

/**
 * A Tensor whose values are truncated to integers (64-bit integer semantics).
 * Negative numbers are truncated toward zero: LongTensor([-1.7]) -> tensor([-1]).
 */
export class LongTensor extends Tensor {
  constructor(data: NestedNumberArray, options: { requires_grad?: boolean } = {}) {
    super(_truncate_nested(data), options);
  }
}
