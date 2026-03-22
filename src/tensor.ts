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

export class Tensor {
  // Auto-generated ID
  public id: number = getNextId();

  // Optional user-defined name
  public name: string | null = null;

  public data: number[];

  public shape: number[];
  public grad_fn: TorchFunction | null = null;
  public grad: Tensor | null = null;

  public requires_grad: boolean;

  constructor(
    data: NestedNumberArray,
    options: { requires_grad?: boolean; name?: string } = {},
    internal_options: { operation?: TorchFunction; shape?: number[] } = {}
  ) {
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
    return this.data.length;
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
    if (this.data.length !== other.data.length) return false;
    for (let i = 0; i < this.data.length; i++) {
      const av = this.data[i],
        bv = other.data[i];
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
}
