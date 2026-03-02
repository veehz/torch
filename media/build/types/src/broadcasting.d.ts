export declare function _broadcast_shape(a_shape: number[], b_shape: number[]): number[];
export declare function _pad_shape(shape: number[], broadcast_shape: number[]): number[];
export declare function _get_original_index(original_shape: number[], new_shape: number[], index: number): number;
export declare function _get_original_index_kernel(original_shape: number[], new_shape: number[], index: number): number;
export declare function _get_original_index_from_transposed_index(original_shape: number[], dim0: number, dim1: number, transposed_index: number): number;
