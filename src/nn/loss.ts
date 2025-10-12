import { Tensor } from "../tensor";

class Loss {}



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