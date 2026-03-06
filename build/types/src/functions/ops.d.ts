import { Tensor } from '../tensor';
import { TorchFunction, BinaryFunction, UnaryFunction } from './base';
export declare class PowInt extends TorchFunction {
    private n;
    protected _forward(a: Tensor, n: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Reshape extends TorchFunction {
    protected _forward(a: Tensor, shape: number[]): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Unsqueeze extends TorchFunction {
    protected _forward(a: Tensor, dim: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Sum extends UnaryFunction {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Mean extends UnaryFunction {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Transpose extends TorchFunction {
    private dim0;
    private dim1;
    protected _forward(a: Tensor, dim0: number, dim1: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Matmul extends BinaryFunction {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
