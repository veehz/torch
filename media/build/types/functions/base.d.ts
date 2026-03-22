import { Tensor } from '../tensor';
type ArgumentType = Tensor | number | number[] | boolean | string;
export declare function resultRequiresGrad(...args: ArgumentType[]): boolean;
declare abstract class TorchFunction {
    id: number;
    opName: string;
    next_functions: TorchFunction[];
    saved_tensors: Tensor[];
    _retained_tensors: Tensor[];
    protected abstract _forward(...args: ArgumentType[]): Tensor;
    protected abstract _backward(dz: Tensor | number): void;
    forward(...args: ArgumentType[]): Tensor;
    backward(dz: Tensor | number): void;
}
declare class NullOp extends TorchFunction {
    protected _forward(..._args: ArgumentType[]): Tensor;
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
