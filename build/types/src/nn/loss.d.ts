import { Tensor } from '../tensor';
declare class Loss {
}
export declare class MSELoss extends Loss {
    constructor();
    forward(input: Tensor, target: Tensor): Tensor;
}
export declare class L1Loss extends Loss {
    constructor();
    forward(input: Tensor, target: Tensor): Tensor;
}
export {};
