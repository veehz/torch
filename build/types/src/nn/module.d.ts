import { Tensor, NestedNumberArray } from '../tensor';
import { Operation } from '../operations/base';
export declare class Parameter extends Tensor {
    constructor(data: NestedNumberArray | Tensor | Parameter, options?: {
        requires_grad?: boolean;
    }, internal_options?: {
        operation?: Operation;
        shape?: number[];
    });
}
export declare abstract class Module {
    private _modules;
    private _parameters;
    constructor();
    private register_parameter;
    private register_module;
    protected register(name: string, value: Parameter | Module): void;
    abstract forward(...args: Tensor[]): Tensor;
    parameters(): Parameter[];
}
export declare class Linear extends Module {
    private weight;
    private bias;
    constructor(in_features: number, out_features: number);
    forward(input: Tensor): Tensor;
}
export declare class ReLU extends Module {
    constructor();
    forward(input: Tensor): Tensor;
}
