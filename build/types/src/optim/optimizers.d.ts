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
