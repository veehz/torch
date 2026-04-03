import { Tensor } from "../tensor";
import { createOperation } from "../functions/registry";

export type Reduction = 'mean' | 'sum' | 'none';

function applyReduction(loss: Tensor, reduction: Reduction): Tensor {
  if (reduction === 'mean') return loss.mean();
  if (reduction === 'sum') return loss.sum();
  return loss;
}

abstract class Loss {
  abstract forward(input: Tensor, target: Tensor): Tensor;
}

export class MSELoss extends Loss {
  private reduction: Reduction;

  constructor(reduction: Reduction = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor) {
    const unreduced = input.sub(target).pow(2);
    return applyReduction(unreduced, this.reduction);
  }
}

export class L1Loss extends Loss {
  private reduction: Reduction;

  constructor(reduction: Reduction = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor) {
    const unreduced = input.sub(target).abs();
    return applyReduction(unreduced, this.reduction);
  }
}

export class BCELoss extends Loss {
  private weight: Tensor | null;
  private reduction: Reduction;

  constructor(weight: Tensor | null = null, reduction: Reduction = 'mean') {
    super();
    this.weight = weight;
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor) {
    const left = target.mul(input.log());
    const right = target.neg().add(1).mul(input.neg().add(1).log());
    let unreduced = left.add(right).neg();
    if (this.weight) {
      unreduced = unreduced.mul(this.weight);
    }
    return applyReduction(unreduced, this.reduction);
  }
}

export class CrossEntropyLoss extends Loss {
  private reduction: Reduction;

  constructor(reduction: Reduction = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor) {
    const op = createOperation('cross_entropy_loss');
    return op.forward(input, target, this.reduction);
  }
}

export class NLLLoss extends Loss {
  private reduction: Reduction;

  constructor(reduction: Reduction = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor) {
    const op = createOperation('nll_loss');
    return op.forward(input, target, this.reduction);
  }
}
