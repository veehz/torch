import { assert } from 'chai';
import { Tensor, no_grad, enable_no_grad, disable_no_grad, is_grad_enabled } from 'torch';

describe('Grad Mode', () => {
  afterEach(() => {
    // Safety: ensure grad mode is always restored after each test
    if (!is_grad_enabled()) disable_no_grad(true);
  });

  describe('is_grad_enabled', () => {
    it('should be enabled by default', () => {
      assert.isTrue(is_grad_enabled());
    });
  });

  describe('no_grad', () => {
    it('should disable gradient tracking inside callback', () => {
      const x = new Tensor([2.0], { requires_grad: true });
      const y = no_grad(() => x.mul(x));

      assert.strictEqual(y.item(), 4.0);
      assert.isFalse(y.requires_grad);
      assert.isNull(y.grad_fn);
    });

    it('should re-enable gradient tracking after callback', () => {
      no_grad(() => {});
      assert.isTrue(is_grad_enabled());
    });

    it('should re-enable gradient tracking even if callback throws', () => {
      try {
        no_grad(() => { throw new Error('boom'); });
      } catch (e) {
        // expected
      }
      assert.isTrue(is_grad_enabled());
    });

    it('should return the value from the callback', () => {
      const result = no_grad(() => 42);
      assert.strictEqual(result, 42);
    });

    it('should nest correctly', () => {
      assert.isTrue(is_grad_enabled());
      no_grad(() => {
        assert.isFalse(is_grad_enabled());
        no_grad(() => {
          assert.isFalse(is_grad_enabled());
        });
        // still disabled after inner no_grad returns
        assert.isFalse(is_grad_enabled());
      });
      assert.isTrue(is_grad_enabled());
    });

    it('should not build computation graph for unary ops', () => {
      const x = new Tensor([2.0], { requires_grad: true });
      const y = no_grad(() => x.exp());

      assert.isFalse(y.requires_grad);
      assert.isNull(y.grad_fn);
    });

    it('should not build computation graph for chained ops', () => {
      const x = new Tensor([3.0], { requires_grad: true });
      const y = no_grad(() => x.mul(x).add(x));

      assert.closeTo(y.item(), 12.0, 1e-6);
      assert.isFalse(y.requires_grad);
      assert.isNull(y.grad_fn);
    });

    it('should not accumulate gradients for operations inside no_grad', () => {
      const x = new Tensor([2.0], { requires_grad: true });

      // Normal forward + backward
      const y = x.mul(x);
      y.backward();
      assert.strictEqual(x.grad.item(), 4.0);

      // Operations inside no_grad should not affect grad
      x.grad = null;
      const z = no_grad(() => x.mul(new Tensor(100.0)));
      // z has no graph, so backward would be a no-op
      assert.isFalse(z.requires_grad);
    });
  });

  describe('enable_no_grad / disable_no_grad', () => {
    it('enable_no_grad should disable grad and return previous state', () => {
      assert.isTrue(is_grad_enabled());
      const prev = enable_no_grad();
      assert.isTrue(prev); // was enabled
      assert.isFalse(is_grad_enabled());
      disable_no_grad(prev);
      assert.isTrue(is_grad_enabled());
    });

    it('should work for pyodide-style usage pattern', () => {
      // Simulate: python side calls enable_no_grad, does work, calls disable_no_grad
      const prev = enable_no_grad();

      const x = new Tensor([5.0], { requires_grad: true });
      const y = x.mul(x);
      assert.isFalse(y.requires_grad);
      assert.isNull(y.grad_fn);

      disable_no_grad(prev);
      assert.isTrue(is_grad_enabled());

      // After restoring, grad should work again
      const a = new Tensor([3.0], { requires_grad: true });
      const b = a.mul(a);
      assert.isTrue(b.requires_grad);
      assert.isNotNull(b.grad_fn);
    });

    it('should handle nested enable/disable correctly', () => {
      const prev1 = enable_no_grad();
      assert.isFalse(is_grad_enabled());

      const prev2 = enable_no_grad();
      assert.isFalse(is_grad_enabled());

      disable_no_grad(prev2); // prev2 is false, still disabled
      assert.isFalse(is_grad_enabled());

      disable_no_grad(prev1); // prev1 is true, now enabled
      assert.isTrue(is_grad_enabled());
    });
  });

  describe('interaction with requires_grad', () => {
    it('tensors created with requires_grad=true keep the flag, but ops do not track', () => {
      const x = new Tensor([2.0], { requires_grad: true });
      // x.requires_grad is an intrinsic property of the tensor, not affected by grad mode
      assert.isTrue(x.requires_grad);

      const y = no_grad(() => {
        // x still says requires_grad=true, but the operation should not build a graph
        return x.mul(x);
      });

      assert.isFalse(y.requires_grad);
    });

    it('backward still works on tensors created before no_grad', () => {
      // Build graph normally
      const x = new Tensor([3.0], { requires_grad: true });
      const y = x.mul(x); // y = 9, dy/dx = 6

      // Enter no_grad for some unrelated work
      no_grad(() => {
        const _ = x.add(new Tensor(100.0)); // no graph built
      });

      // Original graph is intact
      y.backward();
      assert.strictEqual(x.grad.item(), 6.0);
    });
  });
});
