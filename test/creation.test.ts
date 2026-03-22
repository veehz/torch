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
      const t = torch.tensor([
        [1, 2],
        [3, 4]
      ]);
      assert.deepStrictEqual(t.toArray(), [
        [1, 2],
        [3, 4]
      ]);
      assert.deepStrictEqual(t.shape, [2, 2]);
    });

    it('should create a tensor with nested 3D array', () => {
      const t = torch.tensor([
        [
          [1, 2],
          [3, 4]
        ],
        [
          [5, 6],
          [7, 8]
        ]
      ]);
      assert.deepStrictEqual(t.toArray(), [
        [
          [1, 2],
          [3, 4]
        ],
        [
          [5, 6],
          [7, 8]
        ]
      ]);
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
      assert.throws(() =>
        torch.tensor([
          [1, 2],
          [3, 4, 5]
        ])
      );
      assert.throws(() =>
        torch.tensor([
          [
            [1, 2],
            [3, 4]
          ],
          [[5, 6]]
        ])
      );
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
      assert.deepStrictEqual(t.toArray(), [
        [1, 1, 1],
        [1, 1, 1]
      ]);
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
      assert.deepStrictEqual(t.toArray(), [
        [0, 0],
        [0, 0],
        [0, 0]
      ]);
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
      const original = new Tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      const t = torch.ones_like(original);
      assert.deepStrictEqual(t.shape, [2, 3]);
      assert.isTrue(t.toFlatArray().every((v: number) => v === 1));
    });
  });

  describe('zeros_like', () => {
    it('should create a tensor of zeros with the same shape', () => {
      const original = new Tensor([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);
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

  describe('empty', () => {
    it('should create a tensor with the correct shape', () => {
      const t = torch.empty(3, 4);
      assert.deepStrictEqual(t.shape, [3, 4]);
      assert.strictEqual(t.toFlatArray().length, 12);
    });

    it('should accept shape as an array', () => {
      const t = torch.empty([2, 3]);
      assert.deepStrictEqual(t.shape, [2, 3]);
    });
  });

  describe('empty_like', () => {
    it('should create a tensor with the same shape', () => {
      const original = new Tensor([[1, 2, 3], [4, 5, 6]]);
      const t = torch.empty_like(original);
      assert.deepStrictEqual(t.shape, [2, 3]);
    });
  });

  describe('full', () => {
    it('should create a tensor filled with a value', () => {
      const t = torch.full([2, 3], 7);
      assert.deepStrictEqual(t.shape, [2, 3]);
      assert.isTrue(t.toFlatArray().every((v: number) => v === 7));
    });

    it('should create a 1D tensor filled with a value', () => {
      const t = torch.full([5], 3.14);
      assert.deepStrictEqual(t.shape, [5]);
      for (const v of t.toFlatArray()) {
        assert.closeTo(v, 3.14, 1e-6);
      }
    });
  });

  describe('full_like', () => {
    it('should create a tensor filled with a value with the same shape', () => {
      const original = new Tensor([[1, 2], [3, 4]]);
      const t = torch.full_like(original, 42);
      assert.deepStrictEqual(t.shape, [2, 2]);
      assert.isTrue(t.toFlatArray().every((v: number) => v === 42));
    });
  });

  describe('rand_like', () => {
    it('should create a tensor with the same shape and values in [0, 1)', () => {
      const original = new Tensor([[1, 2, 3], [4, 5, 6]]);
      const t = torch.rand_like(original);
      assert.deepStrictEqual(t.shape, [2, 3]);
      for (const v of t.toFlatArray()) {
        assert.isAtLeast(v, 0);
        assert.isBelow(v, 1);
      }
    });
  });

  describe('randn_like', () => {
    it('should create a tensor with the same shape', () => {
      const original = new Tensor([[1, 2], [3, 4]]);
      const t = torch.randn_like(original);
      assert.deepStrictEqual(t.shape, [2, 2]);
      assert.strictEqual(t.toFlatArray().length, 4);
    });
  });

  describe('randint_like', () => {
    it('should create a tensor with the same shape and integer values in [low, high)', () => {
      const original = new Tensor([[1, 2, 3], [4, 5, 6]]);
      const t = torch.randint_like(original, 0, 10);
      assert.deepStrictEqual(t.shape, [2, 3]);
      for (const v of t.toFlatArray()) {
        assert.isAtLeast(v, 0);
        assert.isBelow(v, 10);
        assert.strictEqual(v, Math.floor(v));
      }
    });
  });

  describe('is_tensor', () => {
    it('should return true for tensors', () => {
      const t = torch.tensor([1, 2, 3]);
      assert.isTrue(torch.is_tensor(t));
    });

    it('should return false for non-tensors', () => {
      assert.isFalse(torch.is_tensor(5));
      assert.isFalse(torch.is_tensor([1, 2, 3]));
      assert.isFalse(torch.is_tensor('hello'));
      assert.isFalse(torch.is_tensor(null));
    });
  });

  describe('is_nonzero', () => {
    it('should return true for non-zero scalar tensor', () => {
      assert.isTrue(torch.is_nonzero(torch.tensor(5)));
      assert.isTrue(torch.is_nonzero(torch.tensor(-1)));
      assert.isTrue(torch.is_nonzero(torch.tensor(0.001)));
    });

    it('should return false for zero scalar tensor', () => {
      assert.isFalse(torch.is_nonzero(torch.tensor(0)));
    });

    it('should throw for multi-element tensor', () => {
      assert.throws(() => torch.is_nonzero(torch.tensor([1, 2])));
    });
  });

  describe('numel', () => {
    it('should return the total number of elements', () => {
      assert.strictEqual(torch.numel(torch.tensor([1, 2, 3])), 3);
      assert.strictEqual(torch.numel(torch.tensor([[1, 2], [3, 4]])), 4);
      assert.strictEqual(torch.numel(torch.tensor(5)), 1);
    });
  });

  describe('seed', () => {
    it('random numbers should be different', () => {
      const t1 = torch.rand(5);
      const t2 = torch.rand(5);
      assert.notDeepEqual(t1.toArray(), t2.toArray());
    });

    it('manual_seed should seed the random number generator', () => {
      torch.manual_seed(123);
      const t1 = torch.rand(5);
      torch.manual_seed(123);
      const t2 = torch.rand(5);
      assert.deepStrictEqual(t1.toArray(), t2.toArray());
    });
  });
});
