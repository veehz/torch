import { assert } from 'chai';
import * as torch from '@sourceacademy/torch';

describe('NN Functional', () => {
  describe('Relu', () => {
    it('should forward a tensor', () => {
      const x = new torch.Tensor([1, -2, 3, -4, 5]);
      const result = torch.nn.functional.relu(x);
      assert.deepStrictEqual(Array.from(result.toFlatArray()), [1, 0, 3, 0, 5]);
      assert.deepStrictEqual(result.shape, [5]);
    });
  });

  describe('leaky_relu', () => {
    it('passes positive values unchanged', () => {
      const y = torch.nn.functional.leaky_relu(torch.tensor([1.0, 2.0]));
      assert.deepStrictEqual(y.toArray(), [1.0, 2.0]);
    });

    it('scales negative values by negative_slope', () => {
      const y = torch.nn.functional.leaky_relu(torch.tensor([-2.0]), 0.1);
      assert.closeTo(y.toArray()[0], -0.2, 1e-6);
    });
  });

  describe('max_pool2d', () => {
    it('basic 2x2 pooling', () => {
      const x = torch.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
      const y = torch.nn.functional.max_pool2d(x, 2);
      assert.deepStrictEqual(y.shape, [1, 1, 2, 2]);
      assert.strictEqual(y.toArray()[0][0][0][0], 6);
      assert.strictEqual(y.toArray()[0][0][1][1], 16);
    });

    it('output shape is correct with explicit stride', () => {
      assert.deepStrictEqual(torch.nn.functional.max_pool2d(torch.randn(1, 1, 4, 4), 2, 1).shape, [1, 1, 3, 3]);
    });
  });

  describe('nll_loss', () => {
    it('mean reduction matches expected value', () => {
      const logProbs = torch.tensor([
        [Math.log(0.1), Math.log(0.3), Math.log(0.6)],
        [Math.log(0.7), Math.log(0.2), Math.log(0.1)],
      ]);
      const out = torch.nn.functional.nll_loss(logProbs, torch.tensor([2, 0]));
      assert.closeTo(out.item(), -(Math.log(0.6) + Math.log(0.7)) / 2, 1e-5);
    });

    it('sum reduction', () => {
      const logProbs = torch.tensor([[-1.0, -0.5], [-0.8, -0.3]]);
      const out = torch.nn.functional.nll_loss(logProbs, torch.tensor([1, 0]), 'sum');
      assert.closeTo(out.item(), 1.3, 1e-5);
    });
  });
});
