import { assert } from 'chai';
import { Tensor } from 'torch';

describe('Autograd', () => {
  it('y = x**2, dy/dx at x=2 is 4', () => {
    const x = new Tensor([2.0], { requires_grad: true });
    const y = x.pow(new Tensor([2.0]));
    assert.strictEqual(y.item(), 4.0);
    y.backward();
    assert.strictEqual(x.grad?.item(), 4.0);
  });

  it('y = 2*x, dy/dx at x=2 is 2', () => {
    const x = new Tensor([2.0], { requires_grad: true });
    const y = x.mul(new Tensor(2.0));
    assert.strictEqual(y.item(), 4.0);
    y.backward();
    assert.strictEqual(x.grad?.item(), 2.0);
  });

  it('z = x + y, dz/dx at x=2, y=3 is 1', () => {
    const x = new Tensor([2.0], { requires_grad: true });
    const y = new Tensor([3.0], { requires_grad: true });
    const z = x.add(y);
    assert.strictEqual(z.item(), 5.0);
    z.backward();
    assert.strictEqual(x.grad?.item(), 1.0);
    assert.strictEqual(y.grad?.item(), 1.0);
  });

  it('y = x**2 + 2*x (seperated operations), dy/dx at x=2 is 6', () => {
    const x = new Tensor([2.0], { requires_grad: true });

    const y1 = x.pow(new Tensor(2.0));
    const y2 = x.mul(new Tensor(2.0));

    const y = y1.add(y2);

    assert.strictEqual(y.item(), 8.0);
    assert.strictEqual(y1.item(), 4.0);
    assert.strictEqual(y2.item(), 4.0);

    y.backward();

    assert.strictEqual(y1.grad?.item(), 1.0);
    assert.strictEqual(y2.grad?.item(), 1.0);

    assert.strictEqual(x.grad?.item(), 6.0);
    assert.strictEqual(x.item(), 2.0);
  });

  it('y = x**2 + 2*x (combined operations), dy/dx at x=2 is 6', () => {
    const x = new Tensor([2.0], { requires_grad: true });
    const y = x.pow(new Tensor(2.0)).add(x.mul(new Tensor(2.0)));
    assert.strictEqual(y.item(), 8.0);
    y.backward();
    assert.strictEqual(x.grad?.item(), 6.0);
  });
});
