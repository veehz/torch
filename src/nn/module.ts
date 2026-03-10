import { Module, Parameter } from "./base";
import { rand } from "../creation";
import * as functional from "./functional";
import { Tensor } from "../tensor";

export class Linear extends Module {
  private weight: Parameter;
  private bias: Parameter;

  constructor(in_features: number, out_features: number) {
    super();
    const k = Math.sqrt(1 / in_features);

    this.weight = new Parameter(
      rand([out_features, in_features])
        .mul(2 * k)
        .sub(k)
    );
    this.bias = new Parameter(
      rand([out_features])
        .mul(2 * k)
        .sub(k)
    );

    this.register('weight', this.weight);
    this.register('bias', this.bias);
  }

  forward(input: Tensor) {
    return input.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
}

export class ReLU extends Module {
  constructor() {
    super();
  }

  forward(input: Tensor) {
    return functional.relu(input);
  }
}

export class Sigmoid extends Module {
  constructor() {
    super();
  }

  forward(input: Tensor) {
    return functional.sigmoid(input);
  }
}
