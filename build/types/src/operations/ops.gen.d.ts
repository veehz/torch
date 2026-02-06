import { Tensor } from '../tensor';
import { Operation, BinaryOperation, UnaryOperation } from './base';
export declare class __Left_index__ extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class __Right_index__ extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Add extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Sub extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Mul extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Div extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Pow extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Fmod extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Maximum extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Minimum extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class PowInt extends Operation {
    private n;
    protected _forward(a: Tensor, n: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Log extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Sqrt extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Exp extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Square extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Abs extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Sign extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Neg extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Reciprocal extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Reshape extends Operation {
    protected _forward(a: Tensor, shape: number[]): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Unsqueeze extends Operation {
    protected _forward(a: Tensor, dim: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Sin extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Cos extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Tan extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Sum extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Mean extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Transpose extends Operation {
    private dim0;
    private dim1;
    protected _forward(a: Tensor, dim0: number, dim1: number): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Matmul extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Lt extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Gt extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Le extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Ge extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Eq extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Ne extends BinaryOperation {
    protected _forward(a: Tensor, b: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
