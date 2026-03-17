import { Tensor, NestedNumberArray } from '../tensor';
export declare function tensor(data: NestedNumberArray, requires_grad?: boolean): Tensor;
export declare function ones(...args: number[] | number[][]): Tensor;
export declare function zeros(...args: number[] | number[][]): Tensor;
export declare function ones_like(tensor: Tensor): Tensor;
export declare function zeros_like(tensor: Tensor): Tensor;
