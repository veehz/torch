let globalId = 0;

export const getNextId = () => {
  return globalId++;
};

export const eventBus = new EventTarget();
export const events = {
  TENSOR_BEFORE_BACKWARD: 'tensor.beforeBackward',
  TENSOR_AFTER_BACKWARD: 'tensor.afterBackward',
  OPERATION_BEFORE_FORWARD: 'operation.beforeForward',
  OPERATION_AFTER_FORWARD: 'operation.afterForward',
  OPERATION_BEFORE_BACKWARD: 'operation.beforeBackward',
  OPERATION_AFTER_BACKWARD: 'operation.afterBackward',
  OPERATION_BEFORE_ACCUMULATE_GRAD: 'operation.beforeAccumulateGrad',
  OPERATION_AFTER_ACCUMULATE_GRAD: 'operation.afterAccumulateGrad'
};

export function _numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export function _get_shape_from_args(args: number[] | number[][]): number[] {
  if (Array.isArray(args[0])) {
    return args[0];
  }

  return args as number[];
}
