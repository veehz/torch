import { BinaryFunction, UnaryFunction } from './base';
export declare function BinaryFunctionMixin(operation: (a: number[], b: number[], a_index: number, b_index: number) => number, backward_operations: (a: any, b: any, aFn: any, bFn: any, dz: any) => void, opName?: string | null): typeof BinaryFunction;
export declare function UnaryFunctionMixin(operation: (a: number[], x: number) => number, backward_operations: (a: any, aFn: any, dz: any) => void, opName?: string | null): typeof UnaryFunction;
