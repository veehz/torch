import { TorchFunction, TorchFunctionConstructor } from './base';
export declare function registerOperation(name: string, func: TorchFunctionConstructor): void;
export declare function getOperation(name: string): TorchFunctionConstructor;
export declare function getOperationCache(name: string): TorchFunction;
