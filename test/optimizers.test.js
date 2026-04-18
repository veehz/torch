import { assert } from 'chai';
import * as torch from '@sourceacademy/torch';

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

  describe('Adagrad', () => {
    it('parameters are updated', () => {
      const w = new torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]));
      const opt = new torch.optim.Adagrad([w], 0.1);
      opt.zero_grad();
      w.mul(torch.tensor([1.0, 1.0, 1.0])).sum().backward();
      const before = [...w.toFlatArray()];
      opt.step();
      w.toFlatArray().forEach((v, i) => assert.notEqual(v, before[i]));
    });

    it('step size decreases over time (accumulating squared gradients)', () => {
      const w = new torch.nn.Parameter(torch.tensor([2.0]));
      const opt = new torch.optim.Adagrad([w], 0.1);
      const steps = [];
      for (let i = 0; i < 3; i++) {
        opt.zero_grad();
        w.mul(torch.ones(1)).sum().backward();
        const before = w.item();
        opt.step();
        steps.push(Math.abs(w.item() - before));
      }
      assert.isBelow(steps[1], steps[0]);
      assert.isBelow(steps[2], steps[1]);
    });

    it('weight_decay adds L2 regularization', () => {
      const w1 = new torch.nn.Parameter(torch.tensor([1.0, 2.0]));
      const w2 = new torch.nn.Parameter(torch.tensor([1.0, 2.0]));
      const opt1 = new torch.optim.Adagrad([w1], 0.1, 0.0, 0.0);
      const opt2 = new torch.optim.Adagrad([w2], 0.1, 0.0, 0.01);
      for (const [w, opt] of [[w1, opt1], [w2, opt2]]) {
        opt.zero_grad();
        w.mul(torch.ones(2)).sum().backward();
        opt.step();
      }
      assert.notDeepEqual(w1.toArray(), w2.toArray());
    });

    it('zero_grad clears gradients', () => {
      const w = new torch.nn.Parameter(torch.tensor([1.0, 2.0]));
      const opt = new torch.optim.Adagrad([w], 0.1);
      w.mul(torch.ones(2)).sum().backward();
      assert.isNotNull(w.grad);
      opt.zero_grad();
      assert.isNull(w.grad);
    });
  });
});
