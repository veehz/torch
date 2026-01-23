import { assert } from 'chai';
import { Tensor } from 'torch';

describe('Custom Operations', () => {

  describe('Matmul', () => {
    it('should perform matrix multiplication on 2D tensors', () => {
      const t1 = new Tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      const t2 = new Tensor([
        [7, 8],
        [9, 1],
        [2, 3]
      ]);

      const result = t1.matmul(t2);
      assert.deepStrictEqual(result.shape, [2, 2]);
      assert.deepStrictEqual(Array.from(result.toArray()), [31, 19, 85, 55]);
    });

    it('should perform dot product on 1D tensors', () => {
      const t1 = new Tensor([1, 2, 3]);
      const t2 = new Tensor([4, 5, 6]);

      const result = t1.matmul(t2);
      assert.deepStrictEqual(result.shape, [1]);
      assert.deepStrictEqual(Array.from(result.toArray()), [32]);
    });

    it('should handle batch matrix multiplication', () => {
      const t1 = new Tensor([
        [[62, 50], [7, 53]],
        [[5, 48], [63, 94]]
      ]);
      const t2 = new Tensor([
        [[98, 3], [59, 81]],
        [[79, 74], [41, 98]]
      ]);

      const result = t1.matmul(t2);
      assert.deepStrictEqual(result.shape, [2, 2, 2]);

      const data = Array.from(result.toArray());
      assert.deepStrictEqual(data, [9026, 4236, 3813, 4314, 2363, 5074, 8831, 13874]);
    });
  });

  describe('Transpose', () => {
    it('should transpose a 2D tensor', () => {
      const t = new Tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]);

      const result = t.transpose(0, 1);
      assert.deepStrictEqual(result.shape, [3, 2]);
      assert.deepStrictEqual(Array.from(result.toArray()), [1, 4, 2, 5, 3, 6]);
    });

    it('should transpose dimensions in a 3D tensor', () => {
      const t = new Tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
      ]);

      const result = t.transpose(1, 2);
      assert.deepStrictEqual(result.shape, [2, 2, 2]);
      const data = Array.from(result.toArray());
      assert.deepStrictEqual(data, [1, 3, 2, 4, 5, 7, 6, 8]);
    });
  });

});
