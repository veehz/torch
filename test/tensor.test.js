import { assert } from 'chai';
import * as torch from '@sourceacademy/torch';
import { Tensor } from '@sourceacademy/torch';

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
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3, 4]);
        assert.deepStrictEqual(result.shape, [2, 2]);
      });

      it('should not reshape a tensor if the shape is not compatible', () => {
        const t = new Tensor([1, 2, 3, 4]);
        assert.throws(() => t.reshape([2, 3]), Error);
      });

      it('should reshape a tensor with different dimensions', () => {
        const t = new Tensor([1, 2, 3, 4, 5, 6]);
        const result = t.reshape([2, 3]);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 3]);

        const result2 = t.reshape([3, 2]);
        assert.deepStrictEqual(Array.from(result2.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result2.shape, [3, 2]);

        const result3 = t.reshape([6]);
        assert.deepStrictEqual(Array.from(result3.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result3.shape, [6]);

        const result4 = t.reshape([1, 2, 3]);
        assert.deepStrictEqual(Array.from(result4.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result4.shape, [1, 2, 3]);
      });
    });

    describe('Unsqueeze', () => {
      it('should unsqueeze a tensor', () => {
        const t = new Tensor([1, 2, 3]);
        let result = t.unsqueeze(0);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3]);
        assert.deepStrictEqual(result.shape, [1, 3]);

        let s = new Tensor([1, 2, 3, 4, 5, 6]);
        s = s.reshape([2, 3]);

        result = s.unsqueeze(0);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [1, 2, 3]);

        result = s.unsqueeze(1);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 1, 3]);

        result = s.unsqueeze(2);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 3, 1]);
      });

      it('should unsqueeze a tensor with negative dimension', () => {
        const t = new Tensor([1, 2, 3]);
        let result = t.unsqueeze(-1);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3]);
        assert.deepStrictEqual(result.shape, [3, 1]);

        let s = new Tensor([1, 2, 3, 4, 5, 6]);
        s = s.reshape([2, 3]);

        result = s.unsqueeze(-3);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [1, 2, 3]);

        result = s.unsqueeze(-2);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(result.shape, [2, 1, 3]);

        result = s.unsqueeze(-1);
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 2, 3, 4, 5, 6]);
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

        assert.deepStrictEqual(Array.from(result.toFlatArray()), [30]);
        assert.deepStrictEqual(result.shape, [1]);
      });

      it('should add two 1D tensors of same length', () => {
        const t1 = new Tensor([1, 2, 3]);
        const t2 = new Tensor([4, 5, 6]);
        const result = t1.add(t2);

        assert.deepStrictEqual(Array.from(result.toFlatArray()), [5, 7, 9]);
        assert.deepStrictEqual(result.shape, [3]);
      });
    });

    describe('Multiplication', () => {
      it('should multiply two tensors with same shape, scalar', () => {
        const t1 = new Tensor([10]);
        const t2 = new Tensor([20]);
        const result = t1.mul(t2);

        assert.deepStrictEqual(Array.from(result.toFlatArray()), [200]);
        assert.deepStrictEqual(result.shape, [1]);
      });

      it('should multiply two tensors with same shape, scalar 2', () => {
        const t1 = new Tensor([-1.0604]);
        const t2 = new Tensor([0.756]);
        const result = t1.mul(t2);

        assert.closeTo(Array.from(result.toFlatArray())[0], -0.8016, 0.001);
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

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [10, 20, 30]);
      assert.deepStrictEqual(result.shape, [3]);
    });
    it('should add tensors with different shapes (broadcasting)', () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = t1.add(t2);

      assert.deepStrictEqual(Array.from(result.toFlatArray()), [11, 21, 31]);
      assert.deepStrictEqual(result.shape, [3]);
    });
  });

  describe('Unary Operations', () => {
    describe('Neg', () => {
      it('should negate a tensor', () => {
        const t = new Tensor([1, 2, 3, -4]);
        const result = t.neg();
        assert.deepStrictEqual(Array.from(result.toFlatArray()), [-1, -2, -3, 4]);
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

  describe('Comparison Operations', () => {
    it('le should return 1 where a <= b', () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([2, 2, 1]);
      const result = a.le(b);
      assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 1, 0]);
    });

    it('ge should return 1 where a >= b', () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([2, 2, 1]);
      const result = a.ge(b);
      assert.deepStrictEqual(Array.from(result.toFlatArray()), [0, 1, 1]);
    });

    it('ne should return 1 where a != b', () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([1, 0, 3]);
      const result = a.ne(b);
      assert.deepStrictEqual(Array.from(result.toFlatArray()), [0, 1, 0]);
    });
  });

  describe('Fmod', () => {
    it('should compute element-wise remainder', () => {
      const a = new Tensor([7, -7, 5, 10]);
      const b = new Tensor([3, 3, 2, 4]);
      const result = a.fmod(b);
      assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, -1, 1, 2]);
    });
  });

  describe('Numel', () => {
    it('should return the total number of elements', () => {
      const t = new Tensor([[1, 2, 3], [4, 5, 6]]);
      assert.strictEqual(t.numel(), 6);
    });
  });

  describe('cat', () => {
    describe('1D tensors', () => {
      it('concatenates two 1D tensors along dim 0', () => {
        const a = new Tensor([1, 2, 3]);
        const b = new Tensor([4, 5]);
        const out = torch.cat([a, b]);
        assert.deepStrictEqual(Array.from(out.toFlatArray()), [1, 2, 3, 4, 5]);
        assert.deepStrictEqual(out.shape, [5]);
      });

      it('concatenates three 1D tensors', () => {
        const a = new Tensor([1]);
        const b = new Tensor([2, 3]);
        const c = new Tensor([4, 5, 6]);
        const out = torch.cat([a, b, c]);
        assert.deepStrictEqual(Array.from(out.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(out.shape, [6]);
      });
    });

    describe('2D tensors', () => {
      it('concatenates along dim 0 (row-wise)', () => {
        const a = new Tensor([[1, 2], [3, 4]]);
        const b = new Tensor([[5, 6]]);
        const out = torch.cat([a, b], 0);
        assert.deepStrictEqual(Array.from(out.toFlatArray()), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(out.shape, [3, 2]);
      });

      it('concatenates along dim 1 (column-wise)', () => {
        const a = new Tensor([[1, 2], [3, 4]]);
        const b = new Tensor([[5], [6]]);
        const out = torch.cat([a, b], 1);
        assert.deepStrictEqual(Array.from(out.toFlatArray()), [1, 2, 5, 3, 4, 6]);
        assert.deepStrictEqual(out.shape, [2, 3]);
      });
    });

    describe('3D tensors', () => {
      it('concatenates along dim 0', () => {
        const out = torch.cat([torch.ones([2, 3, 4]), torch.zeros([1, 3, 4])], 0);
        assert.deepStrictEqual(out.shape, [3, 3, 4]);
      });

      it('concatenates along dim 2', () => {
        const out = torch.cat([torch.ones([2, 3, 4]), torch.ones([2, 3, 2])], 2);
        assert.deepStrictEqual(out.shape, [2, 3, 6]);
      });
    });

    describe('negative dim', () => {
      it('dim=-1 is the last dimension', () => {
        const a = new Tensor([[1, 2], [3, 4]]);
        const b = new Tensor([[5], [6]]);
        const pos = torch.cat([a, b], 1);
        const neg = torch.cat([a, b], -1);
        assert.deepStrictEqual(Array.from(neg.toFlatArray()), Array.from(pos.toFlatArray()));
        assert.deepStrictEqual(neg.shape, [2, 3]);
      });

      it('dim=-2 on a 3D tensor', () => {
        const a = torch.ones([2, 3, 4]);
        const b = torch.ones([2, 1, 4]);
        const pos = torch.cat([a, b], 1);
        const neg = torch.cat([a, b], -2);
        assert.deepStrictEqual(neg.shape, pos.shape);
        assert.deepStrictEqual(Array.from(neg.toFlatArray()), Array.from(pos.toFlatArray()));
      });
    });

    describe('autograd', () => {
      it('grad flows back to each input (1D)', () => {
        const x = new Tensor([1, 2, 3], { requires_grad: true });
        const y = new Tensor([4, 5], { requires_grad: true });
        torch.cat([x, y]).sum().backward();
        assert.deepStrictEqual(Array.from(x.grad.toFlatArray()), [1, 1, 1]);
        assert.deepStrictEqual(Array.from(y.grad.toFlatArray()), [1, 1]);
      });

      it('grad flows back to each input (2D, dim 0)', () => {
        const x = new Tensor([[1, 2], [3, 4]], { requires_grad: true });
        const y = new Tensor([[5, 6]], { requires_grad: true });
        torch.cat([x, y], 0).sum().backward();
        assert.deepStrictEqual(Array.from(x.grad.toFlatArray()), [1, 1, 1, 1]);
        assert.deepStrictEqual(Array.from(y.grad.toFlatArray()), [1, 1]);
      });

      it('grad flows back to each input (2D, dim 1)', () => {
        const x = new Tensor([[1, 2], [3, 4]], { requires_grad: true });
        const y = new Tensor([[5], [6]], { requires_grad: true });
        torch.cat([x, y], 1).sum().backward();
        assert.deepStrictEqual(Array.from(x.grad.toFlatArray()), [1, 1, 1, 1]);
        assert.deepStrictEqual(Array.from(y.grad.toFlatArray()), [1, 1]);
      });

      it('non-uniform upstream gradient is sliced correctly', () => {
        const x = new Tensor([[1, 0], [0, 1]], { requires_grad: true });
        const y = new Tensor([[2, 2]], { requires_grad: true });
        const upstream = new Tensor([[1, 2], [3, 4], [5, 6]]);
        torch.cat([x, y], 0).mul(upstream).sum().backward();
        assert.deepStrictEqual(Array.from(x.grad.toFlatArray()), [1, 2, 3, 4]);
        assert.deepStrictEqual(Array.from(y.grad.toFlatArray()), [5, 6]);
      });

      it('grad only flows to tensors with requires_grad=true', () => {
        const x = new Tensor([1, 2, 3], { requires_grad: true });
        const y = new Tensor([4, 5]);
        torch.cat([x, y]).sum().backward();
        assert.deepStrictEqual(Array.from(x.grad.toFlatArray()), [1, 1, 1]);
        assert.isNull(y.grad);
      });

      it('grad flows through three inputs', () => {
        const a = new Tensor([1, 2], { requires_grad: true });
        const b = new Tensor([3], { requires_grad: true });
        const c = new Tensor([4, 5, 6], { requires_grad: true });
        torch.cat([a, b, c]).sum().backward();
        assert.deepStrictEqual(Array.from(a.grad.toFlatArray()), [1, 1]);
        assert.deepStrictEqual(Array.from(b.grad.toFlatArray()), [1]);
        assert.deepStrictEqual(Array.from(c.grad.toFlatArray()), [1, 1, 1]);
      });

      it('cat result can be used in further computation', () => {
        const x = new Tensor([2, 3], { requires_grad: true });
        const y = new Tensor([4], { requires_grad: true });
        torch.cat([x, y]).mul(new Tensor([1, 2, 3])).sum().backward();
        assert.deepStrictEqual(Array.from(x.grad.toFlatArray()), [1, 2]);
        assert.deepStrictEqual(Array.from(y.grad.toFlatArray()), [3]);
      });
    });

    describe('tensor.cat method', () => {
      it('tensor.cat(other) prepends self', () => {
        const a = new Tensor([1, 2, 3]);
        const b = new Tensor([4, 5]);
        assert.deepStrictEqual(Array.from(a.cat(b).toFlatArray()), [1, 2, 3, 4, 5]);
      });

      it('tensor.cat([b, c]) prepends self before b and c', () => {
        const a = new Tensor([1]);
        const b = new Tensor([2, 3]);
        const c = new Tensor([4]);
        assert.deepStrictEqual(Array.from(a.cat([b, c]).toFlatArray()), [1, 2, 3, 4]);
      });

      it('tensor.cat with dim argument', () => {
        const a = new Tensor([[1, 2], [3, 4]]);
        const b = new Tensor([[5, 6]]);
        assert.deepStrictEqual(a.cat(b, 0).shape, [3, 2]);
      });

      it('tensor.cat gradient flows back through self', () => {
        const a = new Tensor([1, 2], { requires_grad: true });
        const b = new Tensor([3, 4], { requires_grad: true });
        a.cat(b).sum().backward();
        assert.deepStrictEqual(Array.from(a.grad.toFlatArray()), [1, 1]);
        assert.deepStrictEqual(Array.from(b.grad.toFlatArray()), [1, 1]);
      });
    });

    describe('aliases (concatenate, concat)', () => {
      it('torch.concatenate produces the same result as torch.cat', () => {
        const a = new Tensor([1, 2, 3]);
        const b = new Tensor([4, 5]);
        assert.deepStrictEqual(
          Array.from(torch.concatenate([a, b]).toFlatArray()),
          Array.from(torch.cat([a, b]).toFlatArray())
        );
      });

      it('torch.concat produces the same result as torch.cat', () => {
        const a = new Tensor([1, 2, 3]);
        const b = new Tensor([4, 5]);
        assert.deepStrictEqual(
          Array.from(torch.concat([a, b]).toFlatArray()),
          Array.from(torch.cat([a, b]).toFlatArray())
        );
      });

      it('tensor.concatenate is an alias for tensor.cat', () => {
        const a = new Tensor([1, 2]);
        const b = new Tensor([3]);
        assert.deepStrictEqual(
          Array.from(a.concatenate(b).toFlatArray()),
          Array.from(a.cat(b).toFlatArray())
        );
      });

      it('tensor.concat is an alias for tensor.cat', () => {
        const a = new Tensor([1, 2]);
        const b = new Tensor([3]);
        assert.deepStrictEqual(
          Array.from(a.concat(b).toFlatArray()),
          Array.from(a.cat(b).toFlatArray())
        );
      });
    });

    describe('error handling', () => {
      it('throws on empty tensor list', () => {
        assert.throws(() => torch.cat([]), /non-empty/);
      });

      it('throws on zero-dimensional tensor', () => {
        const a = new Tensor(1);
        assert.throws(() => torch.cat([a, a]), /zero-dimensional/);
      });

      it('throws on mismatched ndim', () => {
        const a = new Tensor([1, 2]);
        const b = new Tensor([[1, 2]]);
        assert.throws(() => torch.cat([a, b]), /dimensions/);
      });

      it('throws on mismatched non-cat dimension', () => {
        const a = new Tensor([[1, 2, 3]]);
        const b = new Tensor([[1, 2]]);
        assert.throws(() => torch.cat([a, b], 0), /shape/);
      });
    });
  });
});
