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
export declare abstract class Module {
    private _modules;
    private _parameters;
    constructor();
    private register_parameter;
    private register_module;
    protected register(name: string, value: Parameter | Module): void;
    abstract forward(...args: Tensor[]): Tensor;
    parameters(): Parameter[];
    named_parameters(prefix?: string): [string, Parameter][];
}
export declare class Sequential extends Module {
    private _modulesArr;
    constructor(...modules: Module[]);
    append(module: Module): this;
    extend(sequential: Sequential): this;
    insert(index: number, module: Module): this;
    forward(input: Tensor): Tensor;
}
