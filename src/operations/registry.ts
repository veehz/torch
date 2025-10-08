import { OperationConstructor } from './base';

// Only allow registering concrete, constructible Operation classes
const operations = new Map<string, OperationConstructor>();

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
