import { Tensor } from '../tensor';
export declare const opBus: EventTarget;
declare abstract class Operation {
    next_functions: Operation[];
    saved_tensors: Tensor[];
    _retained_tensors: Tensor[];
    protected abstract _forward(...args: (Tensor | number | number[])[]): Tensor;
    protected abstract _backward(dz: Tensor): void;
    forward(...args: (Tensor | number | number[])[]): Tensor;
    backward(dz: Tensor): void;
}
declare class NullOp extends Operation {
    protected _forward(...args: (Tensor | number | number[])[]): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare const nullOp: NullOp;
declare abstract class UnaryOperation extends Operation {
    protected abstract _forward(a: Tensor): Tensor;
    protected abstract _backward(dz: Tensor): void;
}
declare abstract class BinaryOperation extends Operation {
    protected abstract _forward(a: Tensor, b: Tensor): Tensor;
    protected abstract _backward(dz: Tensor): void;
}
export type OperationConstructor = new () => Operation;
export type UnaryOperationConstructor = new () => UnaryOperation;
export type BinaryOperationConstructor = new () => BinaryOperation;
export { Operation, UnaryOperation, BinaryOperation };
export declare class AccumulateGrad extends UnaryOperation {
    variable: Tensor;
    protected _forward(variable: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
