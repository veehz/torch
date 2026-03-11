import { TorchFunction, TorchFunctionConstructor } from './base';

// Only allow registering concrete, constructible Operation classes
const operations = new Map<string, TorchFunctionConstructor>();
const operations_cache = new Map<string, TorchFunction>();

export function registerOperation(name: string, func: TorchFunctionConstructor) {
  operations.set(name, func);
}

function getOperation(name: string): TorchFunctionConstructor {
  const func = operations.get(name);
  if (!func) {
    throw new Error(`Operation '${name}' is not registered.`);
  }
  return func;
}

export function _getAllOperationNames(): Iterable<string> {
  return operations.keys();
}

export function getOperationCache(name: string): TorchFunction {
  const operation = operations_cache.get(name);
  if (!operation) {
    const op = new (getOperation(name))();
    op.opName = name;
    operations_cache.set(name, op);
    return op;
  }
  return operation;
}

/**
 * Create a new operation instance with its opName set.
 */
export function createOperation(name: string): TorchFunction {
  const op = new (getOperation(name))();
  op.opName = name;
  return op;
}
