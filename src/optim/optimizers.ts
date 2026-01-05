import { Optimizer } from "./base";
import { Parameter } from "../nn/module";
import { Tensor } from "../tensor";

export class SGD extends Optimizer {
  private state: Map<Parameter, { velocity: Tensor }> = new Map();
  private lr: number;
  private momentum: number;
  private dampening: number;
  private weight_decay: number;
  private nesterov: boolean;
  private maximize: boolean;
  
  constructor(
    params: Parameter[],
    lr: number = 0.001,
    momentum: number = 0.0,
    dampening: number = 0.0,
    weight_decay: number = 0.0,
    nesterov: boolean = false,
    maximize: boolean = false,
  ) {
    super(params, {});
    this.lr = lr;
    this.momentum = momentum;
    this.dampening = dampening;
    this.weight_decay = weight_decay;
    this.nesterov = nesterov;
    this.maximize = maximize;
  }

  step(): void {
    for (const param of this.params) {
      let g = this.maximize ? param.grad.mul(-1) : param.grad;
      if (this.weight_decay != 0) {
        g = g.add(param.mul(this.weight_decay));
      }

      if (this.momentum != 0) {
        if (this.state.has(param)) {
          let buf = this.state.get(param)!.velocity;
          buf = buf.mul(this.momentum)
          buf = buf.add(g.mul(1 - this.dampening));
          this.state.set(param, { velocity: buf });
        } else {
          this.state.set(param, { velocity: g });
        }

        let buf = this.state.get(param)!.velocity;

        if (this.nesterov) {
          g = g.add(buf.mul(this.momentum));
        } else {
          g = buf;
        }

        this.state.set(param, { velocity: buf });
      }

      // potentially unsafe?
      const newParam = param.sub(g.mul(this.lr));
      param.data = newParam.data;
    }
  }
}
