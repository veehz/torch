import { assert } from 'chai';
import * as torch from 'torch';
import { Tensor } from 'torch';

describe('Functional', () => {
  describe('Addition', () => {
    it('should add two tensors with same shape', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.add(t1, t2);

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [30]);
      assert.deepStrictEqual(result.shape, [1]);
    });

    it('should add tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = torch.add(t1, t2);

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [11, 21, 31]);
      assert.deepStrictEqual(result.shape, [3]);
    });

    it('should add two 1D tensors of same length', () => {
      const t1 = new Tensor([1, 2, 3]);
      const t2 = new Tensor([4, 5, 6]);
      const result = torch.add(t1, t2);

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [5, 7, 9]);
      assert.deepStrictEqual(result.shape, [3]);
    });
  });

  describe('Multiplication', () => {
    it('should multiply two tensors with same shape', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.mul(t1, t2);

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [200]);
      assert.deepStrictEqual(result.shape, [1]);
    });
  });

  describe('should multiply tensors with different shapes (broadcasting)', () => {
    it('should multiply two tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = torch.mul(t1, t2);

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [10, 20, 30]);
      assert.deepStrictEqual(result.shape, [3]);
    });
  });

  describe('Matrix Multiplication', () => {
    it('should multiply two tensors with shape (1)', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.matmul(t1, t2);

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [200]);
      assert.deepStrictEqual(result.shape, []);
    });

    it('should multiply two tensors with 1 dim', () => {
      const t1 = new Tensor([1, 2, 3, 4]);
      const t2 = new Tensor([5, 6, 7, 8]);
      const result = torch.matmul(t1, t2);

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [70]);
      assert.deepStrictEqual(result.shape, []);
    });

    it('should multiply two tensors with shape (2, 3) and (3, 3) with the correct values', () => {
      const t1 = new Tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]);

      const t2 = new Tensor([
        [9, 9, 1],
        [6, 4, 3],
        [5, 5, 6]
      ]);

      const result = torch.matmul(t1, t2);

      const expected = [36, 32, 25, 96, 86, 55];

      assert.deepStrictEqual(Array.from(result.toFlatArray()), expected);
      assert.deepStrictEqual(result.shape, [2, 3]);
    });

    it('should output correct shape', () => {
      function shape_of(shape1, shape2) {
        const tensor1 = torch.randn(shape1);
        const tensor2 = torch.randn(shape2);
        const result = torch.matmul(tensor1, tensor2);
        return result.shape;
      }

      assert.deepStrictEqual(shape_of([3, 4], [4, 5]), [3, 5]);
      assert.deepStrictEqual(shape_of([3, 4], [4]), [3]);
      assert.deepStrictEqual(shape_of([10, 3, 4], [4]), [10, 3]);
      assert.deepStrictEqual(shape_of([10, 3, 4], [10, 4, 5]), [10, 3, 5]);
      assert.deepStrictEqual(shape_of([10, 3, 4], [4, 5]), [10, 3, 5]);
    });
  });

  describe('Transpose', () => {
    it('should transpose a tensor', () => {
      const t = new Tensor([
        [1, 2],
        [3, 4]
      ]);
      const result = t.transpose(0, 1);
      assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 3, 2, 4]);
      assert.deepStrictEqual(result.shape, [2, 2]);
    });
  });

  describe('numel', () => {
    it('should return the total number of elements', () => {
      const t = new Tensor([[1, 2], [3, 4], [5, 6]]);
      assert.strictEqual(torch.numel(t), 6);
    });
  });

  describe('randperm', () => {
    it('should return a permutation of 0..n-1', () => {
      torch.manual_seed(42);
      const t = torch.randperm(5);
      assert.strictEqual(t.numel(), 5);
      const sorted = Array.from(t.toFlatArray()).sort((a, b) => a - b);
      assert.deepStrictEqual(sorted, [0, 1, 2, 3, 4]);
    });
  });

  describe('seed', () => {
    it('should return a seed and make rng deterministic', () => {
      const s = torch.seed();
      assert.isNumber(s);
      const a = torch.rand(3);
      // Re-seed with same seed should produce same values
      torch.manual_seed(s);
      const b = torch.rand(3);
      assert.deepStrictEqual(Array.from(a.toFlatArray()), Array.from(b.toFlatArray()));
    });
  });

  describe('allclose', () => {
    it('returns true for identical tensors', () => {
      const a = new Tensor([1, 2, 3]);
      assert.strictEqual(torch.allclose(a, a), true);
    });

    it('returns true for tensors within default tolerance', () => {
      const a = new Tensor([1.0, 2.0, 3.0]);
      const b = new Tensor([1.0, 2.0, 3.0 + 1e-7]);
      assert.strictEqual(torch.allclose(a, b), true);
    });

    it('returns false for tensors outside default tolerance', () => {
      const a = new Tensor([1.0, 2.0, 3.0]);
      const b = new Tensor([1.0, 2.0, 4.0]);
      assert.strictEqual(torch.allclose(a, b), false);
    });

    it('returns false for tensors of different sizes', () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([1, 2]);
      assert.strictEqual(torch.allclose(a, b), false);
    });

    it('returns false for NaN when equal_nan=false', () => {
      const a = new Tensor([1, NaN, 3]);
      const b = new Tensor([1, NaN, 3]);
      assert.strictEqual(torch.allclose(a, b, 1e-5, 1e-8, false), false);
    });

    it('returns true for NaN when equal_nan=true', () => {
      const a = new Tensor([1, NaN, 3]);
      const b = new Tensor([1, NaN, 3]);
      assert.strictEqual(torch.allclose(a, b, 1e-5, 1e-8, true), true);
    });

    it('tensor.allclose() method matches torch.allclose()', () => {
      const a = new Tensor([1.0, 2.0, 3.0]);
      const b = new Tensor([1.0, 2.0, 3.0 + 1e-7]);
      assert.strictEqual(a.allclose(b), torch.allclose(a, b));
    });
  });

  describe('Softmax', () => {
    it('outputs sum to 1 along specified dim', () => {
      const x = torch.tensor([1.0, 2.0, 3.0]);
      assert.closeTo(torch.softmax(x, 0).sum().item(), 1.0, 1e-6);
    });

    it('tensor.softmax() method works', () => {
      const x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      x.softmax(1).toArray().forEach(row => assert.closeTo(row.reduce((a, b) => a + b, 0), 1.0, 1e-6));
    });

    it('negative dim works', () => {
      const x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      assert.deepStrictEqual(torch.softmax(x, 1).toArray(), torch.softmax(x, -1).toArray());
    });

    it('gradient flows back through softmax (grad of sum = 0)', () => {
      const x = torch.tensor([1.0, 2.0, 3.0], true);
      torch.softmax(x, 0).sum().backward();
      x.grad.toFlatArray().forEach(g => assert.closeTo(g, 0.0, 1e-6));
    });
  });

  describe('Clamp / Clip', () => {
    it('clamps values below min and above max', () => {
      assert.deepStrictEqual(torch.clamp(torch.tensor([-2.0, 0.0, 2.0, 5.0]), 0, 3).toArray(), [0.0, 0.0, 2.0, 3.0]);
    });

    it('torch.clip is an alias for torch.clamp', () => {
      const x = torch.tensor([-1.0, 0.5, 2.0]);
      assert.deepStrictEqual(torch.clamp(x, 0, 1).toArray(), torch.clip(x, 0, 1).toArray());
    });

    it('tensor.clamp() method works', () => {
      assert.deepStrictEqual(torch.tensor([-1.0, 0.5, 2.0]).clamp(0, 1).toArray(), [0.0, 0.5, 1.0]);
    });

    it('gradient is 1 inside range, 0 outside', () => {
      const x = torch.tensor([-1.0, 0.5, 2.0], true);
      torch.clamp(x, 0, 1).sum().backward();
      assert.deepStrictEqual(x.grad.toArray(), [0.0, 1.0, 0.0]);
    });
  });
});
