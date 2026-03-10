import { Module, Parameter } from "./base";
import { Tensor } from "../tensor";
export declare class Linear extends Module {
    private weight;
    private bias;
    constructor(in_features: number, out_features: number);
    forward(input: Tensor): Tensor;
}
export declare class ReLU extends Module {
    constructor();
    forward(input: Tensor): Tensor;
}
export declare class Sigmoid extends Module {
    constructor();
    forward(input: Tensor): Tensor;
}
declare abstract class _ConvNd extends Module {
    weight: Parameter;
    bias: Parameter | null;
    in_channels: number;
    out_channels: number;
    kernel_size: number | number[];
    stride: number | number[];
    padding: number | number[];
    dilation: number | number[];
    groups: number;
    constructor(in_channels: number, out_channels: number, kernel_size: number | number[], stride: number | number[], padding: number | number[], dilation: number | number[], groups: number, bias: boolean, dims: number);
    abstract forward(input: Tensor): Tensor;
}
export declare class Conv1d extends _ConvNd {
    constructor(in_channels: number, out_channels: number, kernel_size: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], groups?: number, bias?: boolean);
    forward(input: Tensor): Tensor;
}
export declare class Conv2d extends _ConvNd {
    constructor(in_channels: number, out_channels: number, kernel_size: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], groups?: number, bias?: boolean);
    forward(input: Tensor): Tensor;
}
export declare class Conv3d extends _ConvNd {
    constructor(in_channels: number, out_channels: number, kernel_size: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], groups?: number, bias?: boolean);
    forward(input: Tensor): Tensor;
}
export {};
