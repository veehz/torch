import { assert } from 'chai';
import { Tensor, no_grad, enable_no_grad, disable_no_grad, is_grad_enabled, nn } from '@sourceacademy/torch';

const { Parameter } = nn;

describe('Parameter', () => {
  afterEach(() => {
    // Safety: ensure grad mode is always restored after each test
    if (!is_grad_enabled()) disable_no_grad(true);
  });

  describe('requires_grad default', () => {
    it('should have requires_grad=true by default (from array)', () => {
      const p = new Parameter([1.0, 2.0, 3.0]);
      assert.isTrue(p.requires_grad);
    });

    it('should have requires_grad=true by default (from Tensor)', () => {
      const t = new Tensor([1.0, 2.0, 3.0]);
      const p = new Parameter(t);
      assert.isTrue(p.requires_grad);
    });

    it('should have a grad_fn (AccumulateGrad) when requires_grad=true', () => {
      const p = new Parameter([1.0, 2.0]);
      assert.isNotNull(p.grad_fn);
    });

    it('should allow explicit requires_grad=false', () => {
      const p = new Parameter([1.0, 2.0], { requires_grad: false });
      assert.isFalse(p.requires_grad);
    });
  });

  describe('no_grad does NOT affect Parameter creation', () => {
    it('Parameter created inside no_grad still has requires_grad=true (from array)', () => {
      const p = no_grad(() => new Parameter([1.0, 2.0, 3.0]));
      assert.isTrue(p.requires_grad);
    });

    it('Parameter created inside no_grad still has requires_grad=true (from Tensor)', () => {
      const t = new Tensor([1.0, 2.0]);
      const p = no_grad(() => new Parameter(t));
      assert.isTrue(p.requires_grad);
    });

    it('Parameter created inside no_grad still has a grad_fn', () => {
      const p = no_grad(() => new Parameter([1.0, 2.0]));
      assert.isNotNull(p.grad_fn);
    });

    it('Parameter created with enable_no_grad/disable_no_grad still has requires_grad=true', () => {
      const prev = enable_no_grad();
      const p = new Parameter([1.0, 2.0, 3.0]);
      disable_no_grad(prev);
      assert.isTrue(p.requires_grad);
    });

    it('grad mode is restored after no_grad block that creates a Parameter', () => {
      no_grad(() => new Parameter([1.0]));
      assert.isTrue(is_grad_enabled());
    });

    it('operations on Parameter inside no_grad do not build a graph, but Parameter retains requires_grad', () => {
      const p = new Parameter([2.0]);
      const result = no_grad(() => p.mul(p));
      // Parameter itself is unaffected by no_grad
      assert.isTrue(p.requires_grad);
      // But the operation result has no grad
      assert.isFalse(result.requires_grad);
      assert.isNull(result.grad_fn);
    });
  });
});
