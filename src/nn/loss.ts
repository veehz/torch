import { Tensor } from "../tensor";
import { createOperation } from "../functions/registry";

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
    const right = target.neg().add(1).mul(input.neg().add(1).log());
    const loss = left.add(right).neg().mean();
    if (this.weight) {
      return loss.mul(this.weight);
    }
    return loss;
  }
}

export class CrossEntropyLoss extends Loss {
  constructor() {
    super();
  }

  forward(input: Tensor, target: Tensor) {
    const op = createOperation('cross_entropy_loss');
    return op.forward(input, target);
  }
}
