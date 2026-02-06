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

  describe('Shape', () => {
    it('should return the shape of the 1D tensor', () => {
      const tensor = new Tensor([10, 20, 30]);
      assert.deepStrictEqual(tensor.shape, [3]);
    });

    it('should return the shape of the 2D tensor', () => {
      const tensor = new Tensor([
        [1, 2, 5],
        [3, 4, 6]
      ]);
      assert.deepStrictEqual(tensor.shape, [2, 3]);
    });

    describe('Reshape', () => {
      it('should reshape a tensor', () => {
        const t = new Tensor([1, 2, 3, 4]);
        const result = t.reshape([2, 2]);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3, 4]);
        assert.deepStrictEqual(result.shape, [2, 2]);
      });

      it('should not reshape a tensor if the shape is not compatible', () => {
        const t = new Tensor([1, 2, 3, 4]);
        assert.throws(() => t.reshape([2, 3]), Error);
      });

      it('should reshape a tensor with different dimensions', () => {
        const t = new Tensor([1, 2, 3, 4, 5, 6]);
        const result = t.reshape([2, 3]);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 3]);

        const result2 = t.reshape([3, 2]);
        assert.deepStrictEqual(Array.from(result2.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result2.shape, [3, 2]);

        const result3 = t.reshape([6]);
        assert.deepStrictEqual(Array.from(result3.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result3.shape, [6]);

        const result4 = t.reshape([1, 2, 3]);
        assert.deepStrictEqual(Array.from(result4.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result4.shape, [1, 2, 3]);
      });
    });

    describe('Unsqueeze', () => {
      it('should unsqueeze a tensor', () => {
        const t = new Tensor([1, 2, 3]);
        let result = t.unsqueeze(0);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3]);
        assert.deepStrictEqual(result.shape, [1, 3]);

        let s = new Tensor([1, 2, 3, 4, 5, 6]);
        s = s.reshape([2, 3]);

        result = s.unsqueeze(0);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [1, 2, 3]);

        result = s.unsqueeze(1);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 1, 3]);

        result = s.unsqueeze(2);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 3, 1]);
      });

      it('should unsqueeze a tensor with negative dimension', () => {
        const t = new Tensor([1, 2, 3]);
        let result = t.unsqueeze(-1);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3]);
        assert.deepStrictEqual(result.shape, [3, 1]);

        let s = new Tensor([1, 2, 3, 4, 5, 6]);
        s = s.reshape([2, 3]);

        result = s.unsqueeze(-3);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [1, 2, 3]);

        result = s.unsqueeze(-2);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 1, 3]);

        result = s.unsqueeze(-1);
        assert.deepStrictEqual(Array.from(result.toArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 3, 1]);
      });
    });
  });
});

describe('Operations', () => {
  describe('Binary Pointwise Operations', () => {
    describe('Addition', () => {
      it('should add two tensors with same shape', () => {
        const t1 = new Tensor([10]);
        const t2 = new Tensor([20]);
        const result = t1.add(t2);

        assert.deepStrictEqual(Array.from(result.toArray()), [30]);
        assert.deepStrictEqual(result.shape, [1]);
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
      it('should multiply two tensors with same shape, scalar', () => {
        const t1 = new Tensor([10]);
        const t2 = new Tensor([20]);
        const result = t1.mul(t2);

        assert.deepStrictEqual(Array.from(result.toArray()), [200]);
        assert.deepStrictEqual(result.shape, [1]);
      });

      it('should multiply two tensors with same shape, scalar 2', () => {
        const t1 = new Tensor([-1.0604]);
        const t2 = new Tensor([0.756]);
        const result = t1.mul(t2);

        assert.closeTo(Array.from(result.toArray())[0], -0.8016, 0.001);
        assert.deepStrictEqual(result.shape, [1]);
      });

      it('should multiply two tensors with the same shape, integers', () => {
        const i1 = [
          [2, 3],
          [5, 7],
          [11, 13]
        ];
        const t1 = new Tensor(i1);

        const i2 = [
          [1, 2],
          [3, 4],
          [5, 6]
        ];
        const t2 = new Tensor(i2);

        const result = t1.mul(t2);
        const expected = [
          [2, 6],
          [15, 28],
          [55, 78]
        ];

        for (let i = 0; i < expected.length; i++) {
          for (let j = 0; j < expected[i].length; j++) {
            assert.closeTo(result.data[i * expected[i].length + j], expected[i][j], 0.001);
          }
        }
      });

      it('should multiply two tensors with the same shape', () => {
        const i1 = [
          [-0.4583, -0.222],
          [-1.3351, -1.0604],
          [-0.4482, -1.316]
        ];
        const t1 = new Tensor(i1);

        const i2 = [
          [0.834, 0.4923],
          [0.7729, 0.756],
          [0.5616, 0.0999]
        ];
        const t2 = new Tensor(i2);

        const result = t1.mul(t2);
        const expected = [
          [-0.3822, -0.1093],
          [-1.0319, -0.8016],
          [-0.2517, -0.1315]
        ];

        for (let i = 0; i < expected.length; i++) {
          for (let j = 0; j < expected[i].length; j++) {
            assert.closeTo(result.data[i * expected[i].length + j], expected[i][j], 0.001);
          }
        }

        assert.deepStrictEqual(result.shape, [3, 2]);
      });
    });
  });

  describe('Operations with Broadcasting', () => {
    it('should multiply two tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = t1.mul(t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [10, 20, 30]);
      assert.deepStrictEqual(result.shape, [3]);
    });
    it('should add tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = t1.add(t2);

      assert.deepStrictEqual(Array.from(result.toArray()), [11, 21, 31]);
      assert.deepStrictEqual(result.shape, [3]);
    });
  });

  describe('Unary Operations', () => {
    describe('Neg', () => {
      it('should negate a tensor', () => {
        const t = new Tensor([1, 2, 3, -4]);
        const result = t.neg();
        assert.deepStrictEqual(Array.from(result.toArray()), [-1, -2, -3, 4]);
        assert.deepStrictEqual(result.shape, [4]);
      });
    });

    describe('Exp', () => {
      const input = [1, 2, 3, -4, 2.5, -6.7];
      const t = new Tensor(input);
      const result = t.exp();
      const expected = input.map(x => Math.exp(x));
      for (let i = 0; i < expected.length; i++) {
        assert.closeTo(result.data[i], expected[i], 0.0001);
      }
      assert.deepStrictEqual(result.shape, [6]);
    });
  });
});
