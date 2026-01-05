import { assert } from 'chai';
import * as torch from 'torch';

const EPS = 1e-6;

describe('Optimizers', () => {
  describe('SGD', () => {
    it('should update parameters', () => {
      const x = new torch.Tensor([1.0], { requires_grad: true });
      const sgd = new torch.optim.SGD([x], 0.01);
      
      const y = x.mul(new torch.Tensor(2.0));
      y.backward();
      
      /**
       * x = 1, grad = 2
       * x_new = x - lr * grad = 1 - 0.01 * 2 = 0.98
       */
      sgd.step();
      const actual = sgd.params[0].data[0];
      const expected = 0.98;
      assert.closeTo(actual, expected, EPS);
    });

    it('should update parameters correctly over multiple steps (quadratic)', () => {
      const x = new torch.Tensor([0.5], { requires_grad: true });
      const sgd = new torch.optim.SGD([x], 0.1);

      let y = x.pow(2);

      /**
       * x = 0.5, grad = 2x = 1
       * x_new = x - lr * grad = 0.5 - 0.1 * 1 = 0.4
       */
      y.backward();
      sgd.step();
      sgd.zero_grad();

      let actual = x.data[0];
      assert.closeTo(actual, 0.4, EPS);

      /**
       * x = 0.4, grad = 2x = 0.8
       * x_new = x - lr * grad = 0.4 - 0.1 * 0.8 = 0.32
       */
      y = x.pow(2);
      y.backward();
      sgd.step();
      
      actual = x.data[0];
      assert.closeTo(actual, 0.32, EPS);
    });
  });

  describe('Adam', () => {
    it('should handle bias correction with constant gradient', () => {
      const x = new torch.Tensor([10.0], { requires_grad: true });
      const adam = new torch.optim.Adam([x], 0.1);

      let y = x.mul(2);
      y.backward();
      adam.step();
      adam.zero_grad();

      assert.closeTo(x.data[0], 9.9, EPS);

      y = x.mul(2);
      y.backward();
      adam.step();

      assert.closeTo(x.data[0], 9.8, EPS);
    });
  });
});
