import { Tensor } from '../tensor';
declare abstract class TorchFunction {
    id: number;
    next_functions: TorchFunction[];
    saved_tensors: Tensor[];
    _retained_tensors: Tensor[];
    protected abstract _forward(...args: (Tensor | number | number[])[]): Tensor;
    protected abstract _backward(dz: Tensor): void;
    forward(...args: (Tensor | number | number[])[]): Tensor;
    backward(dz: Tensor): void;
}
declare class NullOp extends TorchFunction {
    protected _forward(...args: (Tensor | number | number[])[]): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare const nullOp: NullOp;
declare abstract class UnaryFunction extends TorchFunction {
    protected abstract _forward(a: Tensor): Tensor;
    protected abstract _backward(dz: Tensor): void;
}
declare abstract class BinaryFunction extends TorchFunction {
    protected abstract _forward(a: Tensor, b: Tensor): Tensor;
    protected abstract _backward(dz: Tensor): void;
}
export type TorchFunctionConstructor = new () => TorchFunction;
export type UnaryFunctionConstructor = new () => UnaryFunction;
export type BinaryFunctionConstructor = new () => BinaryFunction;
export { TorchFunction, UnaryFunction, BinaryFunction };
export declare class AccumulateGrad extends UnaryFunction {
    variable: Tensor;
    protected _forward(variable: Tensor): Tensor;
    protected _backward(dz: Tensor | number): void;
}
