import { Tensor } from '../tensor';

abstract class Operation {
  abstract forward(...args: (Tensor | number | number[])[]): Tensor;
  abstract backward(dz: Tensor): void;
}

abstract class UnaryOperation extends Operation {
  abstract forward(a: Tensor): Tensor;
  abstract backward(dz: Tensor): void;
}

abstract class BinaryOperation extends Operation {
  abstract forward(a: Tensor, b: Tensor): Tensor;
  abstract backward(dz: Tensor): void;
}

export type OperationConstructor = new () => Operation;
export type UnaryOperationConstructor = new () => UnaryOperation;
export type BinaryOperationConstructor = new () => BinaryOperation;

export { Operation, UnaryOperation, BinaryOperation };
