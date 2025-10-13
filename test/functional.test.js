import { assert } from 'chai';
import * as torch from 'torch';
import { Tensor } from 'torch';

describe('Functional', () => {
  describe('Addition', () => {
    it('should add two tensors with same shape', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.add(t1, t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [30]);
      assert.deepStrictEqual(result.shape, [1]);
    });

    it('should add tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = torch.add(t1, t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [11, 21, 31]);
      assert.deepStrictEqual(result.shape, [3]);
    });

    it('should add two 1D tensors of same length', () => {
      const t1 = new Tensor([1, 2, 3]);
      const t2 = new Tensor([4, 5, 6]);
      const result = torch.add(t1, t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [5, 7, 9]);
      assert.deepStrictEqual(result.shape, [3]);
    });
  });

  describe('Multiplication', () => {
    it('should multiply two tensors with same shape', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.mul(t1, t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [200]);
      assert.deepStrictEqual(result.shape, [1]);
    });
  });

  describe('should multiply tensors with different shapes (broadcasting)', () => {
    it('should multiply two tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = torch.mul(t1, t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [10, 20, 30]);
      assert.deepStrictEqual(result.shape, [3]);
    });
  });

  describe('Matrix Multiplication', () => {
    it('should multiply two tensors with shape (1)', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.matmul(t1, t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [200]);
      assert.deepStrictEqual(result.shape, [1]);
    });

    it('should multiply two tensors with 1 dim', () => {
      const t1 = new Tensor([1, 2, 3, 4]);
      const t2 = new Tensor([5, 6, 7, 8]);
      const result = torch.matmul(t1, t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [70]);
      assert.deepStrictEqual(result.shape, [1]);
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

      assert.deepStrictEqual(Array.from(result.toArray()), expected);
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
      assert.deepStrictEqual(Array.from(result.toArray()), [1, 3, 2, 4]);
      assert.deepStrictEqual(result.shape, [2, 2]);
    });
  });
});
