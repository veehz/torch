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

  describe('LeakyReLU', () => {
    it('positive values pass through unchanged', () => {
      const lr = new torch.nn.LeakyReLU();
      const y = lr.forward(torch.tensor([1.0, 2.0, 3.0]));
      assert.deepStrictEqual(y.toArray(), [1.0, 2.0, 3.0]);
    });

    it('negative values scaled by negative_slope (default 0.01)', () => {
      const lr = new torch.nn.LeakyReLU();
      const y = lr.forward(torch.tensor([-1.0, -2.0]));
      assert.closeTo(y.toArray()[0], -0.01, 1e-6);
      assert.closeTo(y.toArray()[1], -0.02, 1e-6);
    });

    it('custom negative_slope is respected', () => {
      const lr = new torch.nn.LeakyReLU(0.2);
      const y = lr.forward(torch.tensor([-1.0, -2.0]));
      assert.closeTo(y.toArray()[0], -0.2, 1e-6);
      assert.closeTo(y.toArray()[1], -0.4, 1e-6);
    });

    it('gradient for positive input is 1, negative is negative_slope', () => {
      const lr = new torch.nn.LeakyReLU(0.1);
      const x = torch.tensor([1.0, -1.0], true);
      lr.forward(x).sum().backward();
      assert.closeTo(x.grad.toArray()[0], 1.0, 1e-6);
      assert.closeTo(x.grad.toArray()[1], 0.1, 1e-6);
    });
  });

  describe('MaxPool2d', () => {
    it('basic 2x2 pooling on 1x1x4x4 input', () => {
      const pool = new torch.nn.MaxPool2d(2);
      const x = torch.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
      const y = pool.forward(x);
      assert.deepStrictEqual(y.shape, [1, 1, 2, 2]);
      assert.strictEqual(y.toArray()[0][0][0][0], 6);
      assert.strictEqual(y.toArray()[0][0][1][1], 16);
    });

    it('output shape is correct with stride', () => {
      assert.deepStrictEqual(new torch.nn.MaxPool2d(2, 1).forward(torch.randn(1, 1, 4, 4)).shape, [1, 1, 3, 3]);
    });

    it('gradient accumulates at argmax positions', () => {
      const pool = new torch.nn.MaxPool2d(2);
      const x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], true);
      pool.forward(x).sum().backward();
      assert.deepStrictEqual(x.grad.toArray(), [[[[0.0, 0.0], [0.0, 1.0]]]]);
    });

    it('accepts 3D input (no batch dim)', () => {
      assert.deepStrictEqual(new torch.nn.MaxPool2d(2).forward(torch.randn(1, 4, 4)).shape, [1, 2, 2]);
    });
  });

  describe('Dropout', () => {
    it('in eval mode passes input through unchanged', () => {
      const drop = new torch.nn.Dropout(0.5);
      drop.eval();
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      assert.deepStrictEqual(drop.forward(x).toArray(), x.toArray());
    });

    it('p=1 always zeros out in training mode', () => {
      const drop = new torch.nn.Dropout(1.0);
      assert.deepStrictEqual(drop.forward(torch.tensor([1.0, 2.0, 3.0])).toArray(), [0.0, 0.0, 0.0]);
    });

    it('train() / eval() toggle training attribute', () => {
      const drop = new torch.nn.Dropout(0.5);
      assert.isTrue(drop.training);
      drop.eval();
      assert.isFalse(drop.training);
      drop.train();
      assert.isTrue(drop.training);
    });
  });

  describe('Softmax', () => {
    it('output sums to 1 along specified dim', () => {
      const sm = new torch.nn.Softmax(1);
      const y = sm.forward(torch.tensor([[1.0, 2.0, 3.0]]));
      assert.closeTo(y.toArray()[0].reduce((a, b) => a + b, 0), 1.0, 1e-6);
    });

    it('output values are in (0, 1)', () => {
      const y = new torch.nn.Softmax(0).forward(torch.randn(5));
      y.toFlatArray().forEach(v => { assert.isAbove(v, 0); assert.isBelow(v, 1); });
    });
  });

  describe('Flatten', () => {
    it('default flattens all dims except batch (start_dim=1)', () => {
      const y = new torch.nn.Flatten().forward(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]));
      assert.deepStrictEqual(y.shape, [1, 4]);
    });

    it('custom start_dim and end_dim', () => {
      assert.deepStrictEqual(new torch.nn.Flatten(0, 1).forward(torch.randn(2, 3, 4)).shape, [6, 4]);
    });
  });

  describe('training mode propagation', () => {
    it('training=true by default', () => {
      assert.isTrue(new torch.nn.Linear(3, 2).training);
    });

    it('eval() sets training=false', () => {
      const linear = new torch.nn.Linear(3, 2);
      linear.eval();
      assert.isFalse(linear.training);
    });

    it('train() restores training=true', () => {
      const linear = new torch.nn.Linear(3, 2);
      linear.eval();
      linear.train();
      assert.isTrue(linear.training);
    });

    it('eval() propagates to submodules in Sequential', () => {
      const drop = new torch.nn.Dropout(0.5);
      const net = new torch.nn.Sequential(new torch.nn.Linear(2, 2), drop);
      net.eval();
      assert.isFalse(drop.training);
      net.train();
      assert.isTrue(drop.training);
    });
  });
});
