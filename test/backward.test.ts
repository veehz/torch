import { Tensor } from '../src/index';

describe('Autograd', () => {

    test('y = x**2, dy/dx at x=2 is 4', () => {
        const x = new Tensor([2.0], { requires_grad: true });
        const y = x.pow(new Tensor([2.0]));
        expect(y.item()).toBe(4.0);
        y.backward();
        expect(x.grad?.item()).toBe(4.0);
    });

    test('y = 2*x, dy/dx at x=2 is 2', () => {
        const x = new Tensor([2.0], { requires_grad: true });
        const y = x.mul(new Tensor(2.0));
        expect(y.item()).toBe(4.0);
        y.backward();
        expect(x.grad?.item()).toBe(2.0);
    });

    test('z = x + y, dz/dx at x=2, y=3 is 1', () => {
        const x = new Tensor([2.0], { requires_grad: true });
        const y = new Tensor([3.0], { requires_grad: true });
        const z = x.add(y);
        expect(z.item()).toBe(5.0);
        z.backward();
        expect(x.grad?.item()).toBe(1.0);
        expect(y.grad?.item()).toBe(1.0);
    });

  test('y = x**2 + 2*x (seperated operations), dy/dx at x=2 is 6', () => {
    const x = new Tensor([2.0], { requires_grad: true });

    const y1 = x.pow(new Tensor(2.0));
    const y2 = x.mul(new Tensor(2.0));

    const y = y1.add(y2);

    expect(y.item()).toBe(8.0);
    expect(y1.item()).toBe(4.0);
    expect(y2.item()).toBe(4.0);

    y.backward();

    expect(y1.grad?.item()).toBe(1.0);
    expect(y2.grad?.item()).toBe(1.0);

    expect(x.grad?.item()).toBe(6.0);
    expect(x.item()).toBe(2.0);
  });

  test('y = x**2 + 2*x (combined operations), dy/dx at x=2 is 6', () => {
    const x = new Tensor([2.0], { requires_grad: true });
    const y = (x.pow(new Tensor(2.0))).add(x.mul(new Tensor(2.0)));
    expect(y.item()).toBe(8.0);
    y.backward();
    expect(x.grad?.item()).toBe(6.0);
  });
});
