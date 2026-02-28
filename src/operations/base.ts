import { Tensor } from '../tensor';
import { eventBus, getNextId, events } from '../util';

function resultRequiresGrad(...args: (Tensor | number | number[])[]): boolean {
  for (const arg of args) {
    if (arg instanceof Tensor && arg.requires_grad) {
      return true;
    }
  }
  return false;
}

abstract class Operation {
  public id: number = getNextId();
  public next_functions: Operation[] = [];
  public saved_tensors: Tensor[] = [];
  public _retained_tensors: Tensor[] = [];

  protected abstract _forward(...args: (Tensor | number | number[])[]): Tensor;
  protected abstract _backward(dz: Tensor): void;

  forward(...args: (Tensor | number | number[])[]): Tensor {
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
    if (!this.variable.grad) {
      this.variable.grad = new Tensor(new Array(this.variable.dataLength()).fill(0));
    }
    eventBus.dispatchEvent(new CustomEvent(events.OPERATION_ACCUMULATE_GRAD, { detail: { operation: this, dz } }));
    this.variable.grad = this.variable.grad.add(dz);
  }
}
