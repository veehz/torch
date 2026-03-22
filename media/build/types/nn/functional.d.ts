import { Tensor } from "../tensor";
export declare const relu: (a: Tensor | number) => Tensor;
export declare const sigmoid: (a: Tensor | number) => Tensor;
export declare const conv1d: (...args: (Tensor | number | number[] | null)[]) => Tensor;
export declare const conv2d: (...args: (Tensor | number | number[] | null)[]) => Tensor;
export declare const conv3d: (...args: (Tensor | number | number[] | null)[]) => Tensor;
export declare const cross_entropy: (...args: (Tensor | number | number[] | null)[]) => Tensor;
