import { Parameter } from '../nn/module';
export declare abstract class Optimizer {
    params: Parameter[];
    defaults: {
        [key: string]: any;
    };
    constructor(params: Parameter[], defaults: {
        [key: string]: any;
    });
    zero_grad(): void;
    abstract step(): void;
}
