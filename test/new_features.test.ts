import { assert } from 'chai';
import * as torch from 'torch';
import { Tensor, FloatTensor, LongTensor } from 'torch';

// ─── softmax ─────────────────────────────────────────────────────────────────

describe('torch.softmax / tensor.softmax', () => {
  it('outputs sum to 1 along specified dim (1D)', () => {
    const x = torch.tensor([1.0, 2.0, 3.0]);
    const y = torch.softmax(x, 0);
    assert.closeTo(y.sum().item(), 1.0, 1e-6);
  });

  it('outputs sum to 1 along each row (2D, dim=1)', () => {
    const x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]);
    const y = torch.softmax(x, 1);
    assert.deepStrictEqual(y.shape, [2, 3]);
    for (let i = 0; i < 2; i++) {
      const rowSum = (y.toArray() as number[][])[i].reduce((a, b) => a + b, 0);
      assert.closeTo(rowSum, 1.0, 1e-6);
    }
  });

  it('all outputs are in (0, 1)', () => {
    const x = torch.randn(4, 5);
    const y = torch.softmax(x, 1);
    for (const v of y.toFlatArray()) {
      assert.isAbove(v, 0);
      assert.isBelow(v, 1);
    }
  });

  it('tensor.softmax() method works', () => {
    const x = torch.tensor([1.0, 2.0, 3.0]);
    const y = x.softmax(0);
    assert.closeTo(y.sum().item(), 1.0, 1e-6);
  });

  it('gradient flows back through softmax', () => {
    const x = torch.tensor([1.0, 2.0, 3.0], true);
    const y = torch.softmax(x, 0);
    y.sum().backward();
    assert.isDefined(x.grad);
    assert.deepStrictEqual(x.grad.shape, [3]);
    // Gradient of softmax.sum() with respect to any input is 0
    for (const g of x.grad.toFlatArray()) {
      assert.closeTo(g, 0.0, 1e-6);
    }
  });

  it('negative dim works', () => {
    const x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
    const y1 = torch.softmax(x, 1);
    const y2 = torch.softmax(x, -1);
    assert.deepStrictEqual(y1.toArray(), y2.toArray());
  });

  it('nn.Softmax module works', () => {
    const sm = new torch.nn.Softmax(1);
    const x = torch.tensor([[1.0, 2.0, 3.0]]);
    const y = sm.forward(x);
    assert.closeTo((y.toArray() as number[][])[0].reduce((a, b) => a + b, 0), 1.0, 1e-6);
  });
});

// ─── clamp / clip ─────────────────────────────────────────────────────────────

describe('torch.clamp / torch.clip / tensor.clamp', () => {
  it('clamps values below min', () => {
    const x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
    const y = torch.clamp(x, -0.5, 1.5);
    assert.deepStrictEqual(y.toArray(), [-0.5, -0.5, 0.0, 1.0, 1.5]);
  });

  it('clamps values above max', () => {
    const x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]);
    const y = torch.clamp(x, 0.0, 1.0);
    assert.deepStrictEqual(y.toArray(), [0.0, 0.5, 1.0, 1.0, 1.0]);
  });

  it('torch.clip is an alias for torch.clamp', () => {
    const x = torch.tensor([-1.0, 0.5, 2.0]);
    const y1 = torch.clamp(x, 0.0, 1.0);
    const y2 = torch.clip(x, 0.0, 1.0);
    assert.deepStrictEqual(y1.toArray(), y2.toArray());
  });

  it('tensor.clamp() method works', () => {
    const x = torch.tensor([-1.0, 0.5, 2.0]);
    const y = x.clamp(0.0, 1.0);
    assert.deepStrictEqual(y.toArray(), [0.0, 0.5, 1.0]);
  });

  it('gradient is 1 inside range, 0 outside', () => {
    const x = torch.tensor([-1.0, 0.5, 2.0], true);
    const y = torch.clamp(x, 0.0, 1.0);
    y.sum().backward();
    assert.deepStrictEqual(x.grad.toArray(), [0.0, 1.0, 0.0]);
  });

  it('works on 2D tensors', () => {
    const x = torch.tensor([[0.0, 3.0], [-1.0, 0.5]]);
    const y = torch.clamp(x, 0.0, 2.0);
    assert.deepStrictEqual(y.toArray(), [[0.0, 2.0], [0.0, 0.5]]);
  });
});

// ─── nn.LeakyReLU ─────────────────────────────────────────────────────────────

describe('nn.LeakyReLU', () => {
  it('positive values pass through unchanged', () => {
    const lr = new torch.nn.LeakyReLU();
    const x = torch.tensor([1.0, 2.0, 3.0]);
    const y = lr.forward(x);
    assert.deepStrictEqual(y.toArray(), [1.0, 2.0, 3.0]);
  });

  it('negative values scaled by negative_slope (default 0.01)', () => {
    const lr = new torch.nn.LeakyReLU();
    const x = torch.tensor([-1.0, -2.0, -3.0]);
    const y = lr.forward(x);
    assert.closeTo((y.toArray() as number[])[0], -0.01, 1e-6);
    assert.closeTo((y.toArray() as number[])[1], -0.02, 1e-6);
    assert.closeTo((y.toArray() as number[])[2], -0.03, 1e-6);
  });

  it('custom negative_slope is respected', () => {
    const lr = new torch.nn.LeakyReLU(0.2);
    const x = torch.tensor([-1.0, -2.0]);
    const y = lr.forward(x);
    assert.closeTo((y.toArray() as number[])[0], -0.2, 1e-6);
    assert.closeTo((y.toArray() as number[])[1], -0.4, 1e-6);
  });

  it('gradient for positive input is 1', () => {
    const lr = new torch.nn.LeakyReLU(0.1);
    const x = torch.tensor([1.0, 2.0, 3.0], true);
    const y = lr.forward(x);
    y.sum().backward();
    assert.deepStrictEqual(x.grad.toArray(), [1.0, 1.0, 1.0]);
  });

  it('gradient for negative input is negative_slope', () => {
    const lr = new torch.nn.LeakyReLU(0.1);
    const x = torch.tensor([-1.0, -2.0, -3.0], true);
    const y = lr.forward(x);
    y.sum().backward();
    assert.deepStrictEqual(x.grad.toArray(), [0.1, 0.1, 0.1]);
  });
});

// ─── nn.MaxPool2d ─────────────────────────────────────────────────────────────

describe('nn.MaxPool2d', () => {
  it('basic 2x2 pooling on 1x1x4x4 input', () => {
    const pool = new torch.nn.MaxPool2d(2);
    const x = torch.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]);
    const y = pool.forward(x);
    assert.deepStrictEqual(y.shape, [1, 1, 2, 2]);
    const arr = y.toArray() as number[][][][];
    assert.strictEqual(arr[0][0][0][0], 6);
    assert.strictEqual(arr[0][0][0][1], 8);
    assert.strictEqual(arr[0][0][1][0], 14);
    assert.strictEqual(arr[0][0][1][1], 16);
  });

  it('output shape is correct with stride', () => {
    const pool = new torch.nn.MaxPool2d(2, 1);
    const x = torch.randn(1, 1, 4, 4);
    const y = pool.forward(x);
    assert.deepStrictEqual(y.shape, [1, 1, 3, 3]);
  });

  it('output shape is correct with padding', () => {
    const pool = new torch.nn.MaxPool2d(3, 1, 1);
    const x = torch.randn(1, 1, 4, 4);
    const y = pool.forward(x);
    assert.deepStrictEqual(y.shape, [1, 1, 4, 4]);
  });

  it('gradient accumulates at argmax positions', () => {
    const pool = new torch.nn.MaxPool2d(2);
    const x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], true);
    const y = pool.forward(x);
    y.sum().backward();
    assert.deepStrictEqual(x.grad.toArray(), [[[[0.0, 0.0], [0.0, 1.0]]]]);
  });

  it('accepts 3D input (no batch dim)', () => {
    const pool = new torch.nn.MaxPool2d(2);
    const x = torch.randn(1, 4, 4);
    const y = pool.forward(x);
    assert.deepStrictEqual(y.shape, [1, 2, 2]);
  });

  it('multi-channel batch input', () => {
    const pool = new torch.nn.MaxPool2d(2);
    const x = torch.randn(2, 3, 4, 4);
    const y = pool.forward(x);
    assert.deepStrictEqual(y.shape, [2, 3, 2, 2]);
  });
});

// ─── nn.Dropout ───────────────────────────────────────────────────────────────

describe('nn.Dropout', () => {
  it('in eval mode (training=false), passes input through unchanged', () => {
    const drop = new torch.nn.Dropout(0.5);
    drop.eval();
    const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
    const y = drop.forward(x);
    assert.deepStrictEqual(y.toArray(), x.toArray());
  });

  it('p=0 always passes through', () => {
    const drop = new torch.nn.Dropout(0.0);
    const x = torch.tensor([1.0, 2.0, 3.0]);
    const y = drop.forward(x);
    assert.deepStrictEqual(y.toArray(), x.toArray());
  });

  it('p=1 always zeros out', () => {
    const drop = new torch.nn.Dropout(1.0);
    const x = torch.tensor([1.0, 2.0, 3.0]);
    const y = drop.forward(x);
    assert.deepStrictEqual(y.toArray(), [0.0, 0.0, 0.0]);
  });

  it('in training mode, output has same shape', () => {
    const drop = new torch.nn.Dropout(0.5);
    const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
    const y = drop.forward(x);
    assert.deepStrictEqual(y.shape, x.shape);
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

// ─── nn.Flatten ───────────────────────────────────────────────────────────────

describe('nn.Flatten', () => {
  it('default flattens all dims except batch (start_dim=1)', () => {
    const flat = new torch.nn.Flatten();
    const x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]);
    const y = flat.forward(x);
    assert.deepStrictEqual(y.shape, [1, 4]);
  });

  it('custom start_dim and end_dim', () => {
    const flat = new torch.nn.Flatten(0, 1);
    const x = torch.randn(2, 3, 4);
    const y = flat.forward(x);
    assert.deepStrictEqual(y.shape, [6, 4]);
  });

  it('fully flattens with start_dim=0', () => {
    const flat = new torch.nn.Flatten(0, -1);
    const x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]);
    const y = flat.forward(x);
    assert.deepStrictEqual(y.shape, [4]);
  });

  it('works in Sequential', () => {
    const net = new torch.nn.Sequential(
      new torch.nn.Flatten(0, -1)
    );
    const x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
    const y = net.forward(x);
    assert.deepStrictEqual(y.shape, [4]);
  });
});

// ─── nn.NLLLoss ───────────────────────────────────────────────────────────────

describe('nn.NLLLoss', () => {
  it('basic loss computation', () => {
    const loss_fn = new torch.nn.NLLLoss();
    // log-probs: each row sums to 1 before log, so use known values
    const logProbs = torch.tensor([
      [Math.log(0.1), Math.log(0.3), Math.log(0.6)],
      [Math.log(0.7), Math.log(0.2), Math.log(0.1)],
    ]);
    const target = torch.tensor([2, 0]);
    const out = loss_fn.forward(logProbs, target);
    // expected: -(log(0.6) + log(0.7)) / 2
    const expected = -(Math.log(0.6) + Math.log(0.7)) / 2;
    assert.closeTo(out.item(), expected, 1e-5);
  });

  it('gradient flows back to input', () => {
    const loss_fn = new torch.nn.NLLLoss();
    const logProbs = torch.tensor(
      [[-1.0, -2.0, -0.5], [-0.8, -1.2, -0.3]],
      true
    );
    const target = torch.tensor([0, 2]);
    const out = loss_fn.forward(logProbs, target);
    out.backward();
    assert.isDefined(logProbs.grad);
    assert.deepStrictEqual(logProbs.grad.shape, [2, 3]);
    // Gradient at target position should be -1/N
    assert.closeTo((logProbs.grad.toArray() as number[][])[0][0], -0.5, 1e-6);
    assert.closeTo((logProbs.grad.toArray() as number[][])[1][2], -0.5, 1e-6);
    // Gradient at non-target positions should be 0
    assert.closeTo((logProbs.grad.toArray() as number[][])[0][1], 0.0, 1e-6);
    assert.closeTo((logProbs.grad.toArray() as number[][])[0][2], 0.0, 1e-6);
  });

  it('reduction=sum', () => {
    const loss_fn = new torch.nn.NLLLoss('sum');
    const logProbs = torch.tensor([[-1.0, -0.5], [-0.8, -0.3]]);
    const target = torch.tensor([1, 0]);
    const out = loss_fn.forward(logProbs, target);
    // -((-0.5) + (-0.8)) = 1.3
    assert.closeTo(out.item(), 1.3, 1e-5);
  });

  it('reduction=none returns per-sample losses', () => {
    const loss_fn = new torch.nn.NLLLoss('none');
    const logProbs = torch.tensor([[-1.0, -0.5], [-0.8, -0.3]]);
    const target = torch.tensor([1, 0]);
    const out = loss_fn.forward(logProbs, target);
    assert.deepStrictEqual(out.shape, [2]);
    assert.closeTo((out.toArray() as number[])[0], 0.5, 1e-5);
    assert.closeTo((out.toArray() as number[])[1], 0.8, 1e-5);
  });
});

// ─── torch.optim.Adagrad ──────────────────────────────────────────────────────

describe('torch.optim.Adagrad', () => {
  it('parameters are updated', () => {
    const w = new torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]));
    const optimizer = new torch.optim.Adagrad([w], 0.1);
    optimizer.zero_grad();
    w.mul(torch.tensor([1.0, 1.0, 1.0])).sum().backward();
    const before = [...w.toFlatArray()];
    optimizer.step();
    const after = w.toFlatArray();
    for (let i = 0; i < 3; i++) {
      assert.notEqual(after[i], before[i]);
    }
  });

  it('step size decreases over time (accumulating squared gradients)', () => {
    const w = new torch.nn.Parameter(torch.tensor([2.0]));
    const optimizer = new torch.optim.Adagrad([w], 0.1);
    const steps: number[] = [];
    const prev = [2.0];
    for (let i = 0; i < 3; i++) {
      optimizer.zero_grad();
      w.mul(torch.ones(1)).sum().backward();
      const valBefore = w.item();
      optimizer.step();
      steps.push(Math.abs(w.item() - valBefore));
    }
    // Each step should be smaller than the last
    assert.isBelow(steps[1], steps[0]);
    assert.isBelow(steps[2], steps[1]);
  });

  it('weight_decay adds L2 regularization', () => {
    const w1 = new torch.nn.Parameter(torch.tensor([1.0, 2.0]));
    const w2 = new torch.nn.Parameter(torch.tensor([1.0, 2.0]));

    const opt1 = new torch.optim.Adagrad([w1], 0.1);
    const opt2 = new torch.optim.Adagrad([w2], 0.1, 0.0, 0.01);

    for (const [w, opt] of [[w1, opt1], [w2, opt2]] as const) {
      (opt as typeof opt1).zero_grad();
      w.mul(torch.ones(2)).sum().backward();
      (opt as typeof opt1).step();
    }

    assert.notDeepEqual(w1.toArray(), w2.toArray());
  });

  it('zero_grad clears gradients', () => {
    const w = new torch.nn.Parameter(torch.tensor([1.0, 2.0]));
    const optimizer = new torch.optim.Adagrad([w], 0.1);
    w.mul(torch.ones(2)).sum().backward();
    assert.isNotNull(w.grad);
    optimizer.zero_grad();
    assert.isNull(w.grad);
  });
});

// ─── Module train/eval ────────────────────────────────────────────────────────

describe('Module training mode', () => {
  it('training=true by default', () => {
    const linear = new torch.nn.Linear(3, 2);
    assert.isTrue(linear.training);
  });

  it('eval() sets training=false', () => {
    const linear = new torch.nn.Linear(3, 2);
    linear.eval();
    assert.isFalse(linear.training);
  });

  it('train() sets training=true', () => {
    const linear = new torch.nn.Linear(3, 2);
    linear.eval();
    linear.train();
    assert.isTrue(linear.training);
  });

  it('train(false) sets training=false', () => {
    const linear = new torch.nn.Linear(3, 2);
    linear.train(false);
    assert.isFalse(linear.training);
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

// ─── FloatTensor / LongTensor ─────────────────────────────────────────────────

describe('FloatTensor', () => {
  it('is an instance of Tensor', () => {
    const t = new FloatTensor([1.5, 2.5]);
    assert.instanceOf(t, Tensor);
  });

  it('preserves float values unchanged', () => {
    const t = new FloatTensor([1.1, 2.9, -3.7]);
    const data = t.toArray() as number[];
    assert.closeTo(data[0], 1.1, 1e-9);
    assert.closeTo(data[1], 2.9, 1e-9);
    assert.closeTo(data[2], -3.7, 1e-9);
  });

  it('works with nested (2D) data', () => {
    const t = new FloatTensor([[1.5, 2.5], [3.5, 4.5]]);
    assert.deepStrictEqual(t.shape, [2, 2]);
  });

  it('accepts requires_grad option', () => {
    const t = new FloatTensor([1.0, 2.0], { requires_grad: true });
    assert.isTrue(t.requires_grad);
  });

  it('participates in autograd', () => {
    const t = new FloatTensor([2.0, 3.0], { requires_grad: true });
    t.sum().backward();
    assert.deepStrictEqual((t.grad!.toArray() as number[]), [1, 1]);
  });
});

describe('LongTensor', () => {
  it('is an instance of Tensor', () => {
    const t = new LongTensor([1, 2, 3]);
    assert.instanceOf(t, Tensor);
  });

  it('truncates positive floats toward zero', () => {
    const t = new LongTensor([1.1, 1.9, 2.0]);
    assert.deepStrictEqual(t.toArray(), [1, 1, 2]);
  });

  it('truncates negative floats toward zero', () => {
    const t = new LongTensor([-1.1, -1.9, -2.0]);
    assert.deepStrictEqual(t.toArray(), [-1, -1, -2]);
  });

  it('works with nested (2D) data', () => {
    const t = new LongTensor([[1.7, 2.3], [3.9, -4.1]]);
    assert.deepStrictEqual(t.shape, [2, 2]);
    assert.deepStrictEqual(t.toArray(), [[1, 2], [3, -4]]);
  });

  it('accepts integer data unchanged', () => {
    const t = new LongTensor([0, 1, 2, 3]);
    assert.deepStrictEqual(t.toArray(), [0, 1, 2, 3]);
  });

  it('accepts requires_grad option', () => {
    const t = new LongTensor([1.5, 2.5], { requires_grad: true });
    assert.isTrue(t.requires_grad);
    assert.deepStrictEqual(t.toArray(), [1, 2]);
  });

  it('also accessible as torch.LongTensor', () => {
    const t = new torch.LongTensor([1.9, -1.9]);
    assert.instanceOf(t, Tensor);
    assert.deepStrictEqual(t.toArray(), [1, -1]);
  });

  it('also accessible as torch.FloatTensor', () => {
    const t = new torch.FloatTensor([1.5, 2.5]);
    assert.instanceOf(t, Tensor);
  });
});
