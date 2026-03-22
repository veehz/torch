import { Tensor } from '../tensor';
export declare function randn(...args: number[] | number[][]): Tensor;
export declare function rand(...args: number[] | number[][]): Tensor;
export declare function randint(low: number, high: number, shape: number[]): Tensor;
export declare function randperm(n: number): Tensor;
export declare function rand_like(input: Tensor): Tensor;
export declare function randn_like(input: Tensor): Tensor;
export declare function randint_like(input: Tensor, low: number, high: number): Tensor;
