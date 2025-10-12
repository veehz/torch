import { Operation, OperationConstructor } from './base';

// Only allow registering concrete, constructible Operation classes
const operations = new Map<string, OperationConstructor>();
const operations_cache = new Map<string, Operation>();

export function registerOperation(name: string, func: OperationConstructor) {
  operations.set(name, func);
}

export function getOperation(name: string): OperationConstructor {
  const func = operations.get(name);
  if (!func) {
    throw new Error(`Operation '${name}' is not registered.`);
  }
  return func;
}

export function getOperationCache(name: string): Operation {
  const operation = operations_cache.get(name);
  if (!operation) {
    operations_cache.set(name, new (getOperation(name))());
    return operations_cache.get(name)!;
  }
  return operation;
}