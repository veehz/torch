import { Parameter } from "../nn/module";

export abstract class Optimizer {
  params: Parameter[];
  defaults: { [key: string]: any };

  constructor(params: Parameter[], defaults: { [key: string]: any }) {
    this.params = params;
    this.defaults = defaults;
  }

  public zero_grad(): void {
    for (const param of this.params) {
      param.grad = null;
    }
  }

  abstract step(): void;
}
