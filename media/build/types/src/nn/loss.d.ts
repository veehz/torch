import { Tensor } from '../tensor';
declare abstract class Loss {
    abstract forward(input: Tensor, target: Tensor): Tensor;
}
export declare class MSELoss extends Loss {
    constructor();
    forward(input: Tensor, target: Tensor): Tensor;
}
export declare class L1Loss extends Loss {
    constructor();
    forward(input: Tensor, target: Tensor): Tensor;
}
export declare class BCELoss extends Loss {
    private weight;
    constructor(weight?: Tensor | null);
    forward(input: Tensor, target: Tensor): Tensor;
}
export {};
