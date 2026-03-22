import { Tensor } from "../tensor";
export type Reduction = 'mean' | 'sum' | 'none';
declare abstract class Loss {
    abstract forward(input: Tensor, target: Tensor): Tensor;
}
export declare class MSELoss extends Loss {
    private reduction;
    constructor(reduction?: Reduction);
    forward(input: Tensor, target: Tensor): Tensor;
}
export declare class L1Loss extends Loss {
    private reduction;
    constructor(reduction?: Reduction);
    forward(input: Tensor, target: Tensor): Tensor;
}
export declare class BCELoss extends Loss {
    private weight;
    private reduction;
    constructor(weight?: Tensor | null, reduction?: Reduction);
    forward(input: Tensor, target: Tensor): Tensor;
}
export declare class CrossEntropyLoss extends Loss {
    private reduction;
    constructor(reduction?: Reduction);
    forward(input: Tensor, target: Tensor): Tensor;
}
export {};
