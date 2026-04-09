import { Tensor } from '../tensor';
import { NestedNumberArray } from '../tensor';
import { TorchFunction } from '../functions/base';

export class Parameter extends Tensor {
  constructor(
    data: NestedNumberArray | Tensor | Parameter,
    // Default to requires_grad=true
    options: { requires_grad?: boolean } = {
      requires_grad: true
    },
    internal_options: { operation?: TorchFunction; shape?: number[] } = {}
  ) {
    if (data instanceof Tensor) {
      super(data.data, { requires_grad: options.requires_grad }, { shape: data.shape });
    } else {
      super(data, options, internal_options);
    }
  }
}
