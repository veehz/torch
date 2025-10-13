import { assert } from 'chai';
import { Tensor } from 'torch';

describe('Tensor', () => {
  describe('Constructor', () => {
    it('should create a tensor with correct data and shape', () => {
      const tensor = new Tensor([10, 20, 30]);
      assert.deepStrictEqual(tensor.data, [10, 20, 30]);
      assert.deepStrictEqual(tensor.shape, [3]);
    });

    it('should create a tensor with nested array', () => {
      const tensor = new Tensor([
        [1, 2],
        [3, 4]
      ]);
      assert.deepStrictEqual(tensor.data, [1, 2, 3, 4]);
      assert.deepStrictEqual(tensor.shape, [2, 2]);
    });
  });

  describe('Addition', () => {
    it('should add two tensors with same shape', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = t1.add(t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [30]);
      assert.deepStrictEqual(result.shape, [1]);
    });

    it('should add tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = t1.add(t2);

      console.log('shsh', result.shape, result.toArray());

      assert.deepStrictEqual(Array.from(result.toArray()), [11, 21, 31]);
      assert.deepStrictEqual(result.shape, [3]);
    });

    it('should add two 1D tensors of same length', () => {
      const t1 = new Tensor([1, 2, 3]);
      const t2 = new Tensor([4, 5, 6]);
      const result = t1.add(t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [5, 7, 9]);
      assert.deepStrictEqual(result.shape, [3]);
    });
  });

  describe('Multiplication', () => {
    it('should multiply two tensors with same shape', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = t1.mul(t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [200]);
      assert.deepStrictEqual(result.shape, [1]);
    });
  });

  describe('should multiply tensors with different shapes (broadcasting)', () => {
    it('should multiply two tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = t1.mul(t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [10, 20, 30]);
      assert.deepStrictEqual(result.shape, [3]);
    });
  });

  describe('Matrix Multiplication', () => {
    it('should multiply two tensors with dim 1', () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = t1.matmul(t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [200]);
      assert.deepStrictEqual(result.shape, [1]);
    });

    it('should multiply two tensors with the correct values', () => {
      const t1 = new Tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]);

      const t2 = new Tensor([
        [9, 9, 1],
        [6, 4, 3],
        [5, 5, 6]
      ]);

      const result = t1.matmul(t2);

      const expected = [36, 32, 25, 96, 86, 55];

      assert.deepStrictEqual(Array.from(result.toArray()), expected);
      assert.deepStrictEqual(result.shape, [2, 3]);
    });
  });
});
