import { TorchFunction } from './functions/base';
export type TypedArray = Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;
export type NestedNumberArray = number | TypedArray | NestedNumberArray[];
/**
 * A shared backing store for tensor data.
 * Multiple tensors (views) may reference the same TensorStorage instance.
 * Mutating `data` on the TensorStorage is visible to all sharing tensors.
 */
export declare class TensorStorage {
    data: number[];
    constructor(data: number[]);
}
export declare class Tensor {
    id: number;
    name: string | null;
    private _storage;
    private _offset;
    /**
     * Returns the flat, contiguous data for this tensor.
     *
     * Fast path (non-view): returns the storage array directly — no allocation.
     * View path: materialises a contiguous slice — one allocation per call,
     * so callers inside tight loops should cache the result: `const d = t.data`.
     */
    get data(): number[];
    /**
     * Sets the tensor's data.
     *
     * Non-view (offset=0, storage covers exactly this tensor's numel):
     *   replaces the shared storage's data array in-place — all other views
     *   sharing the same TensorStorage immediately see the new values.
     *
     * View (offset≠0 or storage is larger than this tensor):
     *   copies `values` element-by-element into the shared storage at the
     *   correct offset — the original tensor and sibling views are updated.
     */
    set data(values: number[]);
    shape: number[];
    grad_fn: TorchFunction | null;
    grad: Tensor | null;
    requires_grad: boolean;
    constructor(data: NestedNumberArray, options?: {
        requires_grad?: boolean;
        name?: string;
    }, internal_options?: {
        operation?: TorchFunction;
        shape?: number[];
        /** For internal view construction only — share an existing storage. */
        _storage?: TensorStorage;
        /** Byte offset into _storage (in elements). */
        _offset?: number;
    });
    size(dim?: number): number | number[];
    toArray_(): void;
    toFlatArray(): number[];
    toArray(): NestedNumberArray;
    toString(): string;
    dataLength(): number;
    private _executeUnaryOp;
    private _executeBinaryOp;
    private _executeOpRaw;
    item(): number;
    detach(): Tensor;
    detach_(): void;
    zero_(): void;
    private is_retain_grad;
    retain_grad(): void;
    backward(grad?: Tensor | null): void;
    /**
     * Returns a view of this tensor along dimension 0.
     *
     * The returned tensor shares the same underlying TensorStorage — mutations
     * to either tensor (via zero_(), the data setter, or the optimizer) are
     * immediately visible in the other.
     *
     * Supports negative indices (e.g. index(-1) is the last row).
     *
     * Note: the view does not carry a grad_fn; autograd does not propagate
     * through index() at this time.
     */
    index(i: number): Tensor;
    add(other: Tensor | number): Tensor;
    sub(other: Tensor | number): Tensor;
    mul(other: Tensor | number): Tensor;
    div(other: Tensor | number): Tensor;
    pow(other: Tensor | number): Tensor;
    fmod(other: Tensor | number): Tensor;
    maximum(other: Tensor | number): Tensor;
    minimum(other: Tensor | number): Tensor;
    log(): Tensor;
    sqrt(): Tensor;
    exp(): Tensor;
    square(): Tensor;
    abs(): Tensor;
    sign(): Tensor;
    neg(): Tensor;
    reciprocal(): Tensor;
    nan_to_num(): Tensor;
    reshape(shape: number[]): Tensor;
    flatten(start_dim?: number, end_dim?: number): Tensor;
    squeeze(dim: number): Tensor;
    unsqueeze(dim: number): Tensor;
    expand(sizes: number[]): Tensor;
    sin(): Tensor;
    cos(): Tensor;
    tan(): Tensor;
    sum(dim?: number | number[], keepdim?: boolean): Tensor;
    mean(dim?: number | number[], keepdim?: boolean): Tensor;
    max(dim?: number | number[], keepdim?: boolean): Tensor;
    min(dim?: number | number[], keepdim?: boolean): Tensor;
    transpose(dim0: number, dim1: number): Tensor;
    matmul(other: Tensor): Tensor;
    lt(other: Tensor | number): Tensor;
    gt(other: Tensor | number): Tensor;
    le(other: Tensor | number): Tensor;
    ge(other: Tensor | number): Tensor;
    eq(other: Tensor | number): Tensor;
    ne(other: Tensor | number): Tensor;
    allclose(other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean): boolean;
    numel(): number;
    sigmoid(): Tensor;
    relu(): Tensor;
}
