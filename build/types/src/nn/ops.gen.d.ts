import { Tensor } from '../tensor';
import { UnaryOperation } from '../operations/base';
export declare class Relu extends UnaryOperation {
    private cache;
    forward(a: Tensor): Tensor;
    backward(dz: Tensor): void;
}
