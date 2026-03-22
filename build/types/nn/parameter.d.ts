import { Tensor } from '../tensor';
import { NestedNumberArray } from '../tensor';
import { TorchFunction } from '../functions/base';
export declare class Parameter extends Tensor {
    constructor(data: NestedNumberArray | Tensor | Parameter, options?: {
        requires_grad?: boolean;
    }, internal_options?: {
        operation?: TorchFunction;
        shape?: number[];
    });
}
