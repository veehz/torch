import { Optimizer } from "./base";
import { Parameter } from "../nn/module";
import { Tensor } from "../tensor";
import { zeros_like } from "../creation";

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
      if (this.weight_decay !== 0) {
        g = g.add(param.mul(this.weight_decay));
      }

      if (this.momentum !== 0) {
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

export class Adam extends Optimizer {
  private state: Map<Parameter, {
    m: Tensor,
    v: Tensor,
    vmax: Tensor
  }> = new Map();

  private step_count: number = 0;
  private lr: number;
  private beta1: number;
  private beta2: number;
  private eps: number;
  private weight_decay: number;
  private amsgrad: boolean;
  private maximize: boolean;

  constructor(
    params: Parameter[],
    lr: number = 0.001,
    betas: [number, number] = [0.9, 0.999],
    eps: number = 1e-8,
    weight_decay: number = 0.0,
    amsgrad: boolean = false,
    maximize: boolean = false,
  ) {
    super(params, {});
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.weight_decay = weight_decay;
    this.amsgrad = amsgrad;
    this.maximize = maximize;
  }

  step(): void {
    this.step_count += 1;
    for (const param of this.params) {
      let grad = this.maximize ? param.grad.mul(-1) : param.grad;

      if (this.weight_decay !== 0) {
        grad = grad.add(param.mul(this.weight_decay));
      }

      // Initialize
      if (!this.state.has(param)) {
        this.state.set(param, {
          m: zeros_like(param),
          v: zeros_like(param),
          vmax: zeros_like(param),
        });
      }

      const state = this.state.get(param)!;

      state.m = state.m.mul(this.beta1).add(grad.mul(1 - this.beta1));
      state.v = state.v.mul(this.beta2).add(grad.mul(grad).mul(1 - this.beta2));

      const biasCorrection1 = 1 - Math.pow(this.beta1, this.step_count);
      const biasCorrection2 = 1 - Math.pow(this.beta2, this.step_count);

      let vhat: Tensor;
      const mhat = state.m.div(biasCorrection1);
      if (this.amsgrad) {
        state.vmax = state.vmax.maximum(state.v);
        vhat = state.vmax.div(biasCorrection2);
      } else {
        vhat = state.v.div(biasCorrection2);
      }

      const update = mhat.div(vhat.sqrt().add(this.eps)).mul(this.lr);

      const newParam = param.sub(update);
      param.data = newParam.data;
    }
  }
}