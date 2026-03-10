/**
 * Global gradient computation mode.
 *
 * When disabled, all operations behave as if no input requires grad,
 * so no computation graph is built and no tensors are saved for backward.
 */
let _grad_enabled = true;

export function is_grad_enabled(): boolean {
  return _grad_enabled;
}

/**
 * Disable gradient computation. Returns the previous state
 * so it can be restored later with `disable_no_grad(prev)`.
 */
export function enable_no_grad(): boolean {
  const prev = _grad_enabled;
  _grad_enabled = false;
  return prev;
}

/**
 * Restore gradient computation to a previous state
 * (as returned by `enable_no_grad`).
 */
export function disable_no_grad(prev: boolean): void {
  _grad_enabled = prev;
}

/**
 * Execute `fn` with gradient computation disabled.
 * The previous grad mode is always restored, even if `fn` throws.
 */
export function no_grad<T>(fn: () => T): T {
  const prev = enable_no_grad();
  try {
    return fn();
  } finally {
    disable_no_grad(prev);
  }
}
