import { Tensor } from '../tensor';
import { Operation, BinaryOperation, UnaryOperation } from './base';
export declare class Add extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Sub extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Mul extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Div extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Pow extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Fmod extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class PowInt extends Operation {
    private cache;
    forward(a: Tensor, n: number): Tensor;
    backward(dz: Tensor): void;
}
export declare class Log extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Sqrt extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Exp extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Abs extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Sign extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Neg extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Reciprocal extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Reshape extends Operation {
    private cache;
    forward(a: Tensor, shape: number[]): Tensor;
    backward(dz: Tensor): void;
}
export declare class Unsqueeze extends Operation {
    private cache;
    forward(a: Tensor, dim: number): Tensor;
    backward(dz: Tensor): void;
}
export declare class Sin extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Cos extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Tan extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Sum extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Mean extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Transpose extends Operation {
    cache: [Tensor, number, number];
    forward(a: Tensor, dim0: number, dim1: number): Tensor;
    backward(dz: Tensor): void;
}
export declare class Matmul extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Lt extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Gt extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Le extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Ge extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Eq extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
export declare class Ne extends BinaryOperation {
    private cache;
    forward(a: Tensor, b: Tensor): Tensor;
    backward(dz: Tensor): void;
}
