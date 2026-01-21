import { Tensor } from "../tensor";

abstract class Loss {
  abstract forward(input: Tensor, target: Tensor): Tensor;
}

export class MSELoss extends Loss {
  constructor() {
    super();
  }

  forward(input: Tensor, target: Tensor) {
    return input.sub(target).pow(2).mean();
  }
}

export class L1Loss extends Loss {
  constructor() {
    super();
  }

  forward(input: Tensor, target: Tensor) {
    return input.sub(target).abs().mean();
  }
}

export class BCELoss extends Loss {
  private weight: Tensor | null;

  constructor(weight: Tensor | null = null) {
    super();
    this.weight = weight;
  }

  forward(input: Tensor, target: Tensor) {
    const left = target.mul(input.log());
    console.log("input", input);
    console.log("input.log()", input.log());
    console.log("target", target);
    console.log("LEFT", left);
    const right = target.neg().add(1).mul(input.neg().add(1).log());
    console.log("RIGHT", right);
    const loss = left.add(right).neg().mean();
    if (this.weight) {
      return loss.mul(this.weight);
    }
    return loss;
  }
}