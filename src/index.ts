export { Tensor } from './tensor';
export type { TypedArray, NestedNumberArray } from './tensor';
export { no_grad, enable_no_grad, disable_no_grad, is_grad_enabled } from './grad_mode';

export { TorchFunction, AccumulateGrad } from './functions/base';
export * from './functions/ops';
export * from './functions/functional';

export * from './creation/index';

export * as nn from './nn/index';

export * as optim from './optim/index';

export { seed, manual_seed } from './prng';

export { eventBus, events } from './util';

export { export_, ExportedProgram } from './export';
export type { GraphNode, InputSpec, OutputSpec, GraphSignature } from './export';

import { Tensor } from './tensor';

export function is_tensor(obj: unknown): boolean {
  return obj instanceof Tensor;
}

export function is_nonzero(input: Tensor): boolean {
  if (input.numel() !== 1) {
    throw new Error(
      `Boolean value of Tensor with more than one element is ambiguous`
    );
  }
  return input.item() !== 0;
}

export function numel(input: Tensor): number {
  return input.numel();
}
