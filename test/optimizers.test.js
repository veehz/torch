import { assert } from 'chai';
import * as torch from 'torch';

describe('Optimizers', () => {
  describe('SGD', () => {
    it('should update parameters', () => {
      const x = new torch.Tensor([1.0], { requires_grad: true });
      const y = x.mul(new torch.Tensor(2.0));
      y.backward();

      const sgd = new torch.optim.SGD([x], 0.01);
      sgd.step();
      const actual = sgd.params[0].data[0];
      const expected = 0.98;
      assert.ok(Math.abs(actual - expected) < 1e-6);
    });
  });
});
