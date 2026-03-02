import { Tensor } from '../tensor';
import { UnaryOperation } from '../operations/base';
export declare class Relu extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
export declare class Sigmoid extends UnaryOperation {
    protected _forward(a: Tensor): Tensor;
    protected _backward(dz: Tensor): void;
}
