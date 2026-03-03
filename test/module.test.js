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

    it('should backward a linear with correct shape', () => {
      const linear = new torch.nn.Linear(10, 20);
      const input = new torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const output = linear.forward(input).sum();
      output.backward();
      assert.deepStrictEqual(linear.weight.grad.shape, [20, 10]);
      assert.deepStrictEqual(linear.bias.grad.shape, [20]);
    });

    it('should backward a linear with correct values', () => {
      const linear = new torch.nn.Linear(2, 3);
      linear.weight.data = new torch.Tensor([[1, 2], [3, 4], [5, 6]]);
      linear.bias.data = new torch.Tensor([1, 2, 3]);
      const input = new torch.Tensor([1, 2]);
      const output = linear.forward(input).sum();
      output.backward();
      assert.deepStrictEqual(linear.weight.grad.data, [1, 2, 1, 2, 1, 2]);
      assert.deepStrictEqual(linear.bias.grad.data, [1, 1, 1]);
    });
  });

  describe('Sequential', () => {
    it('should forward the correct shape', () => {
      const model = new torch.nn.Sequential(
        new torch.nn.Linear(10, 20),
        new torch.nn.ReLU(),
        new torch.nn.Linear(20, 30)
      );

      const input = new torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const output = model.forward(input);
      assert.deepStrictEqual(output.shape, [30]);
    });

    it('should append a module', () => {
      const model = new torch.nn.Sequential();
      model.append(new torch.nn.Linear(10, 20));
      model.append(new torch.nn.ReLU());
      model.append(new torch.nn.Linear(20, 30));

      const input = new torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const output = model.forward(input);
      assert.deepStrictEqual(output.shape, [30]);
    });

    it('should extend with modules', () => {
      const model = new torch.nn.Sequential(new torch.nn.Linear(10, 20), new torch.nn.ReLU());
      const model2 = new torch.nn.Sequential(new torch.nn.Linear(20, 30), new torch.nn.ReLU());
      model.extend(model2);

      const input = new torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const output = model.forward(input);
      assert.deepStrictEqual(output.shape, [30]);
    });

    it('should insert a module', () => {
      const model = new torch.nn.Sequential(
        new torch.nn.Linear(10, 20),
        new torch.nn.Linear(30, 40)
      );
      model.insert(1, new torch.nn.Linear(20, 30));

      const input = new torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const output = model.forward(input);
      assert.deepStrictEqual(output.shape, [40]);
    });
  });
});
