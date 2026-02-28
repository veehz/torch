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
  OPERATION_ACCUMULATE_GRAD: 'operation.accumulateGrad',
}