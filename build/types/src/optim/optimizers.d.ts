import { Optimizer } from './base';
import { Parameter } from '../nn/module';
export declare class SGD extends Optimizer {
    private state;
    private lr;
    private momentum;
    private dampening;
    private weight_decay;
    private nesterov;
    private maximize;
    constructor(params: Parameter[], lr?: number, momentum?: number, dampening?: number, weight_decay?: number, nesterov?: boolean, maximize?: boolean);
    step(): void;
}
export declare class Adam extends Optimizer {
    private state;
    private step_count;
    private lr;
    private beta1;
    private beta2;
    private eps;
    private weight_decay;
    private amsgrad;
    private maximize;
    constructor(params: Parameter[], lr?: number, betas?: [number, number], eps?: number, weight_decay?: number, amsgrad?: boolean, maximize?: boolean);
    step(): void;
}
