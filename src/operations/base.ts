import { Tensor } from '../tensor';

export const opBus = new EventTarget();

abstract class Operation {
  public next_functions: Operation[] = [];
  public saved_tensors: Tensor[] = [];
  public _retained_tensors: Tensor[] = [];

  protected abstract _forward(...args: (Tensor | number | number[])[]): Tensor;
  protected abstract _backward(dz: Tensor): void;

  forward(...args: (Tensor | number | number[])[]): Tensor {
    const result = this._forward(...args);
    opBus.dispatchEvent(new CustomEvent('forward', { detail: { operation: this, args, result } }));
    return result;
  }

  backward(dz: Tensor): void {
    opBus.dispatchEvent(new CustomEvent('backward', { detail: { operation: this, dz } }));
    for (const x of this._retained_tensors) {
      if (!x.grad) {
        x.grad = new Tensor(new Array(x.dataLength()).fill(0));
      }
      x.grad = x.grad.add(dz);
    }
    this._backward(dz);
  }
}

class NullOp extends Operation {
  protected _forward(...args: (Tensor | number | number[])[]): Tensor {
    throw new Error('NullOp should not be called');
  }
  protected _backward(dz: Tensor): void {
    return;
  }
}

export const nullOp = new NullOp();

abstract class UnaryOperation extends Operation {
  protected abstract _forward(a: Tensor): Tensor;
  protected abstract _backward(dz: Tensor): void;
}

abstract class BinaryOperation extends Operation {
  protected abstract _forward(a: Tensor, b: Tensor): Tensor;
  protected abstract _backward(dz: Tensor): void;
}

export type OperationConstructor = new () => Operation;
export type UnaryOperationConstructor = new () => UnaryOperation;
export type BinaryOperationConstructor = new () => BinaryOperation;

export { Operation, UnaryOperation, BinaryOperation };

export class AccumulateGrad extends UnaryOperation {
  public variable: Tensor;

  protected _forward(variable: Tensor): Tensor {
    this.variable = variable;
    return variable;
  }

  protected _backward(dz: Tensor): void {
    if(!this.variable.grad) {
      this.variable.grad = new Tensor(new Array(this.variable.dataLength()).fill(0));
    }
    this.variable.grad = this.variable.grad.add(dz);
  }
}
