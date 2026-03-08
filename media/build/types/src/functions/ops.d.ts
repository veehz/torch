import { Tensor } from '../tensor';
import { TorchFunction, BinaryFunction } from './base';
export declare class PowInt extends TorchFunction {
    private n;
    protected _forward(a: Tensor, n: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Reshape extends TorchFunction {
    protected _forward(a: Tensor, shape: number[]): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Squeeze extends TorchFunction {
    protected _forward(a: Tensor, dim?: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Unsqueeze extends TorchFunction {
    protected _forward(a: Tensor, dim: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Expand extends TorchFunction {
    protected _forward(a: Tensor, expanded_shape: number[]): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare const Sum: new () => TorchFunction;
export declare const Mean: new () => TorchFunction;
export declare const Max: new () => TorchFunction;
export declare const Min: new () => TorchFunction;
export declare class Transpose extends TorchFunction {
    private dim0;
    private dim1;
    protected _forward(a: Tensor, dim0: number, dim1: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Matmul extends BinaryFunction {
    private shape;
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
