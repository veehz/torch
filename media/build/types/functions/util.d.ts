export declare function _get_strides(shape: number[]): number[];
export declare function _unravel_index(index: number, strides: number[]): number[];
export declare function _ravel_index(coords: number[], strides: number[]): number;
export declare function _get_reduction_shape(shape: number[], dim?: number | number[], keepdim?: boolean): number[];
