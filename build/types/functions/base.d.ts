import { Tensor } from '../tensor';
export declare function resultRequiresGrad(...args: (Tensor | number | number[] | boolean)[]): boolean;
declare abstract class TorchFunction {
    id: number;
    opName: string;
    next_functions: TorchFunction[];
    saved_tensors: Tensor[];
    _retained_tensors: Tensor[];
    protected abstract _forward(...args: (Tensor | number | number[] | boolean)[]): Tensor;
    protected abstract _backward(dz: Tensor | number): void;
    forward(...args: (Tensor | number | number[] | boolean)[]): Tensor;
    backward(dz: Tensor | number): void;
}
declare class NullOp extends TorchFunction {
    protected _forward(..._args: (Tensor | number | number[])[]): Tensor;
    protected _backward(_dz: Tensor): void;
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
