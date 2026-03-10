export declare function is_grad_enabled(): boolean;
/**
 * Disable gradient computation. Returns the previous state
 * so it can be restored later with `disable_no_grad(prev)`.
 */
export declare function enable_no_grad(): boolean;
/**
 * Restore gradient computation to a previous state
 * (as returned by `enable_no_grad`).
 */
export declare function disable_no_grad(prev: boolean): void;
/**
 * Execute `fn` with gradient computation disabled.
 * The previous grad mode is always restored, even if `fn` throws.
 */
export declare function no_grad<T>(fn: () => T): T;
