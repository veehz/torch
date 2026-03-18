import { assert } from 'chai';
import * as torch from 'torch';
import { Tensor } from 'torch';

describe('Creation Functions', () => {
  describe('tensor', () => {
    it('should create a scalar tensor', () => {
      const t = torch.tensor(5);
      assert.deepStrictEqual(t.toArray(), 5);
      assert.deepStrictEqual(t.shape, []);
    });

    it('should create a tensor with 1D array', () => {
      const t = torch.tensor([1, 2, 3]);
      assert.deepStrictEqual(t.toArray(), [1, 2, 3]);
      assert.deepStrictEqual(t.shape, [3]);
    });

    it('should create a tensor with nested 2D array', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      assert.deepStrictEqual(t.toArray(), [[1, 2], [3, 4]]);
      assert.deepStrictEqual(t.shape, [2, 2]);
    });

    it('should create a tensor with nested 3D array', () => {
      const t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      assert.deepStrictEqual(t.toArray(), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      assert.deepStrictEqual(t.shape, [2, 2, 2]);
    });

    it('should create empty tensors', () => {
      const t = torch.tensor([]);
      assert.deepStrictEqual(t.toArray(), []);
      assert.deepStrictEqual(t.shape, [0]);

      const t2 = torch.tensor([[]]);
      assert.deepStrictEqual(t2.toArray(), [[]]);
      assert.deepStrictEqual(t2.shape, [1, 0]);

      const t3 = torch.tensor([[], []]);
      assert.deepStrictEqual(t3.toArray(), [[], []]);
      assert.deepStrictEqual(t3.shape, [2, 0]);
    });

    it('should not create tensor with inconsistent shape', () => {
      assert.throws(() => torch.tensor([[1, 2], [3]]));
      assert.throws(() => torch.tensor([1, [2, 3]]));
      assert.throws(() => torch.tensor([[1, 2], [3, 4, 5]]));
      assert.throws(() => torch.tensor([[[1, 2], [3, 4]], [[5, 6]]]));
    });
  });
  describe('ones', () => {
    it('should create a 1D tensor of ones', () => {
      const t = torch.ones(5);
      assert.deepStrictEqual(t.toArray(), [1, 1, 1, 1, 1]);
      assert.deepStrictEqual(t.shape, [5]);
    });

    it('should create a 2D tensor of ones', () => {
      const t = torch.ones(2, 3);
      assert.deepStrictEqual(t.toArray(), [[1, 1, 1], [1, 1, 1]]);
      assert.deepStrictEqual(t.shape, [2, 3]);
    });

    it('should create a 3D tensor of ones', () => {
      const t = torch.ones(2, 3, 4);
      assert.deepStrictEqual(t.shape, [2, 3, 4]);
      assert.strictEqual(t.toFlatArray().length, 24);
      assert.isTrue(t.toFlatArray().every((v: number) => v === 1));
    });

    it('should accept shape as an array', () => {
      const t = torch.ones([3, 2]);
      assert.deepStrictEqual(t.shape, [3, 2]);
      assert.isTrue(t.toFlatArray().every((v: number) => v === 1));
    });
  });

  describe('zeros', () => {
    it('should create a 1D tensor of zeros', () => {
      const t = torch.zeros(4);
      assert.deepStrictEqual(t.toArray(), [0, 0, 0, 0]);
      assert.deepStrictEqual(t.shape, [4]);
    });

    it('should create a 2D tensor of zeros', () => {
      const t = torch.zeros(3, 2);
      assert.deepStrictEqual(t.toArray(), [[0, 0], [0, 0], [0, 0]]);
      assert.deepStrictEqual(t.shape, [3, 2]);
    });

    it('should accept shape as an array', () => {
      const t = torch.zeros([2, 4]);
      assert.deepStrictEqual(t.shape, [2, 4]);
      assert.isTrue(t.toFlatArray().every((v: number) => v === 0));
    });
  });

  describe('ones_like', () => {
    it('should create a tensor of ones with the same shape', () => {
      const original = new Tensor([[1, 2, 3], [4, 5, 6]]);
      const t = torch.ones_like(original);
      assert.deepStrictEqual(t.shape, [2, 3]);
      assert.isTrue(t.toFlatArray().every((v: number) => v === 1));
    });
  });

  describe('zeros_like', () => {
    it('should create a tensor of zeros with the same shape', () => {
      const original = new Tensor([[1, 2], [3, 4], [5, 6]]);
      const t = torch.zeros_like(original);
      assert.deepStrictEqual(t.shape, [3, 2]);
      assert.isTrue(t.toFlatArray().every((v: number) => v === 0));
    });
  });

  describe('rand', () => {
    it('should create a tensor with the correct shape', () => {
      const t = torch.rand(3, 4);
      assert.deepStrictEqual(t.shape, [3, 4]);
      assert.strictEqual(t.toFlatArray().length, 12);
    });

    it('should create values in [0, 1)', () => {
      const t = torch.rand(10);
      for (const v of t.toFlatArray()) {
        assert.isAtLeast(v, 0);
        assert.isBelow(v, 1);
      }
    });
  });

  describe('randn', () => {
    it('should create a tensor with the correct shape', () => {
      const t = torch.randn(2, 5);
      assert.deepStrictEqual(t.shape, [2, 5]);
      assert.strictEqual(t.toFlatArray().length, 10);
    });
  });

  describe('randint', () => {
    it('should create a tensor with the correct shape', () => {
      const t = torch.randint(0, 10, [3, 3]);
      assert.deepStrictEqual(t.shape, [3, 3]);
      assert.strictEqual(t.toFlatArray().length, 9);
    });

    it('should create integer values in [low, high)', () => {
      const t = torch.randint(5, 15, [20]);
      for (const v of t.toFlatArray()) {
        assert.isAtLeast(v, 5);
        assert.isBelow(v, 15);
        assert.strictEqual(v, Math.floor(v));
      }
    });
  });

  describe('linspace', () => {
    it('should create evenly spaced values', () => {
      const t = torch.linspace(0, 1, 5);
      const expected = [0, 0.25, 0.5, 0.75, 1.0];
      assert.deepStrictEqual(t.shape, [5]);
      const arr = t.toFlatArray();
      for (let i = 0; i < expected.length; i++) {
        assert.closeTo(arr[i], expected[i], 1e-6);
      }
    });

    it('should handle negative range', () => {
      const t = torch.linspace(-1, 1, 3);
      const arr = t.toFlatArray();
      assert.closeTo(arr[0], -1, 1e-6);
      assert.closeTo(arr[1], 0, 1e-6);
      assert.closeTo(arr[2], 1, 1e-6);
    });
  });

  describe('arange', () => {
    it('should create a range of values', () => {
      const t = torch.arange(0, 5);
      assert.deepStrictEqual(t.toArray(), [0, 1, 2, 3, 4]);
    });

    it('should support custom step', () => {
      const t = torch.arange(0, 10, 2);
      assert.deepStrictEqual(t.toArray(), [0, 2, 4, 6, 8]);
    });

    it('should support negative step values with appropriate range', () => {
      const t = torch.arange(1, 5, 0.5);
      assert.deepStrictEqual(t.shape, [8]);
      const arr = t.toFlatArray();
      assert.closeTo(arr[0], 1.0, 1e-6);
      assert.closeTo(arr[7], 4.5, 1e-6);
    });
  });
});
