import { assert } from 'chai';
import * as torch from 'torch';

describe('Module', () => {
  describe('Linear', () => {
    it('should create a linear module', () => {
      const linear = new torch.nn.Linear(10, 20);
      assert.deepStrictEqual(linear.parameters().length, 2);
      assert.deepStrictEqual(linear.parameters()[0].shape, [20, 10]);
      assert.deepStrictEqual(linear.parameters()[1].shape, [20]);
    });

    it('should forward a tensor', () => {
      const linear = new torch.nn.Linear(10, 20);
      const input = new torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const output = linear.forward(input);
      assert.deepStrictEqual(output.shape, [20]);
    });

    it('should forward a tensor (2D)', () => {
      // example from https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
      const m = new torch.nn.Linear(20, 30);
      const input = torch.randn(128, 20);
      const output = m.forward(input);
      assert.deepStrictEqual(output.shape, [128, 30]);
    });
  });
});
