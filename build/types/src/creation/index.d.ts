import { Tensor } from '../tensor';
export declare function linspace(start: number, end: number, steps: number): Tensor;
export declare function ones(...args: number[] | number[][]): Tensor;
export declare function zeros(...args: number[] | number[][]): Tensor;
export declare function randn(...args: number[] | number[][]): Tensor;
export declare function rand(...args: number[] | number[][]): Tensor;
export declare function randint(low: number, high: number, shape: number[]): Tensor;
