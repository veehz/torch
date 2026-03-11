import { TorchFunction, TorchFunctionConstructor } from './base';
export declare function registerOperation(name: string, func: TorchFunctionConstructor): void;
export declare function _getAllOperationNames(): Iterable<string>;
export declare function getOperationCache(name: string): TorchFunction;
/**
 * Create a new operation instance with its opName set.
 */
export declare function createOperation(name: string): TorchFunction;
