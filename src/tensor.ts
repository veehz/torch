import { _get_original_index } from './broadcasting';
import { Operation } from './operations/base';
import { getOperation } from './operations/registry';

/*
 * TODO:
 * - Add support for Textures to be stored in Tensors
 */

type TypedArray =
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
  data: number[];
  _shape: number[];
  operation: Operation | null = null;
  public grad: Tensor | null = null;

  requires_grad: boolean;

  constructor(
    data: NestedNumberArray,
    options: { requires_grad?: boolean } = {},
    internal_options: { operation?: Operation; shape?: number[] } = {}
  ) {
    this.data = _flatten(data);
    this.requires_grad = options.requires_grad ?? false;

    this._shape = internal_options.shape ?? _get_shape(data);
    this.operation = internal_options.operation ?? null;
  }

  // TODO: Somehow having a shape of [] will have a weird error:
  // TypeError: Cannot read properties of undefined (reading 'length')
  // when running kernel (something to do with constants?)
  // so a little hack to return [1] when the shape is []
  get shape(): number[] {
    return this._shape.length === 0 ? [1] : this._shape;
    // return this._shape;
  }

  set shape(shape: number[]) {
    this._shape = shape;
  }

  private _executeUnaryOp(opName: string): Tensor {
    const operation = new (getOperation(opName))();
    return operation.forward(this);
  }

  private _executeBinaryOp(opName: string, other: Tensor | number): Tensor {
    if (typeof other == 'number') {
      other = new Tensor(other);
    }
    const operation = new (getOperation(opName))();
    return operation.forward(this, other);
  }

  private _executeOpRaw(opName: string, ...args: any[]): Tensor {
    const operation = new (getOperation(opName))();
    return operation.forward(this, ...args);
  }

  item(): number {
    if (this.data.length !== 1) {
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
    this.operation = null;
  }

  backward(grad?: Tensor | null): void {
    if (!this.requires_grad) {
      // If this tensor does not require gradients, stop propagation.
      // TODO: check pytorch behaviour
      return;
    }

    if (!grad) {
      if (this.data.length !== 1) {
        throw new Error('Gradient is required for non-scalar tensors');
      }

      grad = new Tensor(1);
    }

    if (!this.grad) {
      this.grad = new Tensor(Array(this.data.length).fill(0));
    }

    // Add grad to this.grad
    for (let i = 0; i < grad.data.length; i++) {
      this.grad.data[_get_original_index(this.shape, grad.shape, i)] += grad.data[i];
    }

    if (this.operation) {
      // Propagate only the incoming local gradient, not the accumulated one,
      // to avoid double-counting when a tensor receives gradients from
      // multiple downstream paths.
      // this.operation.backward(grad);
      this.operation.backward(this.grad);
    }
  }

  // operations

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

  sum(): Tensor {
    return this._executeUnaryOp('sum');
  }

  mean(): Tensor {
    return this._executeUnaryOp('mean');
  }

  pow(other: Tensor | number): Tensor {
    return this._executeBinaryOp('pow', other);
  }

  transpose(dim0: number, dim1: number): Tensor {
    return this._executeOpRaw('transpose', dim0, dim1);
  }

  matmul(other: Tensor): Tensor {
    return this._executeBinaryOp('matmul', other);
  }

  fmod(other: Tensor | number): Tensor {
    return this._executeBinaryOp('fmod', other);
  }

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

  sqrt(): Tensor {
    return this._executeUnaryOp('sqrt');
  }

  sign(): Tensor {
    return this._executeUnaryOp('sign');
  }

  abs(): Tensor {
    return this._executeUnaryOp('abs');
  }

  log(): Tensor {
    return this._executeUnaryOp('log');
  }
}
