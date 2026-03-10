import { Tensor } from '../tensor';
import { TorchFunction, BinaryFunction, UnaryFunction } from './base';
export declare function BinaryFunctionMixin(operation: (a: number[], b: number[], a_index: number, b_index: number) => number, backward_operations: (a: any, b: any, aFn: any, bFn: any, dz: any) => void, opName?: string | null): typeof BinaryFunction;
export declare function UnaryFunctionMixin(operation: (a: number[], x: number) => number, backward_operations: (a: any, aFn: any, dz: any) => void, opName?: string | null): typeof UnaryFunction;
export declare function ReductionFunctionMixin(init_val: number, reduce_op: (acc: number, val: number) => number, backward_operations: (a: Tensor, restored_dz: Tensor, dim: number | number[], keepdim: boolean) => Tensor, opName?: string | null, finalize_op?: (acc: number, count: number) => number): new () => TorchFunction;
