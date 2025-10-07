import { _get_original_index } from "./broadcasting";
import { Operation } from "./operations/function_base";
import { getOperation } from "./operations/registry";
import { TensorBase } from "./tensor_base";

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

type NestedNumberArray = number | TypedArray | NestedNumberArray[];

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
    return data.flatMap((item) => _flatten(item));
  } else if (ArrayBuffer.isView(data)) {
    return Array.from(data);
  } else {
    return [data];
  }
}

export class Tensor extends TensorBase {
  data: number[];
  _shape: number[];
  operation: any = null;
  public grad: Tensor | null = null;

  requires_grad: boolean;

  constructor(
    data: NestedNumberArray,
    options: { requires_grad?: boolean } = {},
    internal_options: { operation?: Operation, shape?: number[] } = {}
  ) {
    super();
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

  private _executeBinaryOp(opName: string, other: Tensor): Tensor {
    const operation = new (getOperation(opName))();
    return operation.forward(this, other);
  }

  add(other: Tensor): Tensor {
    return this._executeBinaryOp("add", other);
  }

  sub(other: Tensor): Tensor {
    return this._executeBinaryOp("sub", other);
  }

  mul(other: Tensor): Tensor {
    return this._executeBinaryOp("mul", other);
  }

  div(other: Tensor): Tensor {
    return this._executeBinaryOp("div", other);
  }

  sum(): Tensor {
    return this._executeUnaryOp("sum");
  }

  pow(other: Tensor): Tensor {
    return this._executeBinaryOp("pow", other);
  }

  log(): Tensor {
    return this._executeUnaryOp("log");
  }

  item(): number {
    if (this.data.length !== 1) {
      throw new Error("Tensor.item() is only valid for scalars");
    }
    return this.data[0];
  }

  backward(grad?: Tensor | null): void {
    if (!this.requires_grad) {
      // If this tensor does not require gradients, stop propagation.
      return;
    }

    if (!grad) {
      if (this.data.length !== 1) {
        throw new Error("Gradient is required for non-scalar tensors");
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
}
