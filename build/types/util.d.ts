export declare const getNextId: () => number;
export declare const eventBus: EventTarget;
export declare const events: {
    TENSOR_BEFORE_BACKWARD: string;
    TENSOR_AFTER_BACKWARD: string;
    OPERATION_BEFORE_FORWARD: string;
    OPERATION_AFTER_FORWARD: string;
    OPERATION_BEFORE_BACKWARD: string;
    OPERATION_AFTER_BACKWARD: string;
    OPERATION_BEFORE_ACCUMULATE_GRAD: string;
    OPERATION_AFTER_ACCUMULATE_GRAD: string;
};
export declare function _numel(shape: number[]): number;
export declare function _get_shape_from_args(args: number[] | number[][]): number[];
