import { Tensor } from '../tensor';

export const opBus = new EventTarget();

abstract class Operation {
  protected abstract _forward(...args: (Tensor | number | number[])[]): Tensor;
  protected abstract _backward(dz: Tensor): void;

  forward(...args: (Tensor | number | number[])[]): Tensor {
    const result = this._forward(...args);
    opBus.dispatchEvent(new CustomEvent('forward', { detail: { operation: this, args, result } }));
    return result;
  }

  backward(dz: Tensor): void {
    opBus.dispatchEvent(new CustomEvent('backward', { detail: { operation: this, dz } }));
    this._backward(dz);
  }
}

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
