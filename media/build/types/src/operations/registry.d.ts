import { Operation, OperationConstructor } from './base';
export declare function registerOperation(name: string, func: OperationConstructor): void;
export declare function getOperation(name: string): OperationConstructor;
export declare function getOperationCache(name: string): Operation;
