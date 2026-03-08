import { _get_original_index, _pad_shape, _unbroadcast } from '../broadcasting';
import { zeros_like } from '../creation';
import { Tensor } from '../tensor';
import { eventBus, getNextId, events } from '../util';

function resultRequiresGrad(...args: (Tensor | number | number[] | boolean)[]): boolean {
  for (const arg of args) {
    if (arg instanceof Tensor && arg.requires_grad) {
      return true;
    }
  }
  return false;
}

abstract class TorchFunction {
  public id: number = getNextId();
  public next_functions: TorchFunction[] = [];
  public saved_tensors: Tensor[] = [];
  public _retained_tensors: Tensor[] = [];

  protected abstract _forward(...args: (Tensor | number | number[] | boolean)[]): Tensor;
  protected abstract _backward(dz: Tensor): void;

  forward(...args: (Tensor | number | number[] | boolean)[]): Tensor {
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

  backward(dz: Tensor): void {
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
  protected _forward(...args: (Tensor | number | number[])[]): Tensor {
    throw new Error('NullOp should not be called');
  }
  protected _backward(dz: Tensor): void {
    return;
  }
}

export const nullOp = new NullOp();

abstract class UnaryFunction extends TorchFunction {
  protected abstract _forward(a: Tensor): Tensor;
  protected abstract _backward(dz: Tensor): void;
}

abstract class BinaryFunction extends TorchFunction {
  protected abstract _forward(a: Tensor, b: Tensor): Tensor;
  protected abstract _backward(dz: Tensor): void;
}

export type TorchFunctionConstructor = new () => TorchFunction;
export type UnaryFunctionConstructor = new () => UnaryFunction;
export type BinaryFunctionConstructor = new () => BinaryFunction;

export { TorchFunction, UnaryFunction, BinaryFunction };

export class AccumulateGrad extends UnaryFunction {
  public variable: Tensor;

  protected _forward(variable: Tensor): Tensor {
    this.variable = variable;
    return variable;
  }

  protected _backward(dz: Tensor | number): void {
    if (!this.variable.grad) {
      this.variable.grad = zeros_like(this.variable);
    }
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz } }));
    if(typeof dz === "number") {
      this.variable.grad = this.variable.grad.add(dz);
    } else {
      const unbroadcasted_dz = _unbroadcast(dz.shape, this.variable.shape, dz.data);
      this.variable.grad = this.variable.grad.add(new Tensor(unbroadcasted_dz, {}, { shape: this.variable.shape }));
    }
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz } }));
  }
}
