import { TorchFunction, TorchFunctionConstructor } from './base';

// Only allow registering concrete, constructible Operation classes
const operations = new Map<string, TorchFunctionConstructor>();
const operations_cache = new Map<string, TorchFunction>();

export function registerOperation(name: string, func: TorchFunctionConstructor) {
  operations.set(name, func);
}

export function getOperation(name: string): TorchFunctionConstructor {
  const func = operations.get(name);
  if (!func) {
    throw new Error(`Operation '${name}' is not registered.`);
  }
  return func;
}

export function getOperationCache(name: string): TorchFunction {
  const operation = operations_cache.get(name);
  if (!operation) {
    operations_cache.set(name, new (getOperation(name))());
    return operations_cache.get(name)!;
  }
  return operation;
}
