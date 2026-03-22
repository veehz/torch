import { assert } from 'chai';
import * as torch from 'torch';
import { Tensor } from 'torch';

describe('Linear bias parameter', () => {
  it('should create Linear with bias=true by default', () => {
    const linear = new torch.nn.Linear(4, 3);
    assert.strictEqual(linear.parameters().length, 2);
    assert.deepStrictEqual(linear.parameters()[0].shape, [3, 4]); // weight
    assert.deepStrictEqual(linear.parameters()[1].shape, [3]);    // bias
  });

  it('should create Linear with bias=false', () => {
    const linear = new torch.nn.Linear(4, 3, false);
    assert.strictEqual(linear.parameters().length, 1);
    assert.deepStrictEqual(linear.parameters()[0].shape, [3, 4]); // weight only
    assert.isNull(linear.bias);
  });

  it('should forward without bias', () => {
    const linear = new torch.nn.Linear(3, 2, false);
    const input = new Tensor([1, 2, 3]);
    const output = linear.forward(input);
    assert.deepStrictEqual(output.shape, [2]);
  });

  it('should backward without bias', () => {
    const linear = new torch.nn.Linear(3, 2, false);
    const input = new Tensor([1, 2, 3]);
    const output = linear.forward(input).sum();
    output.backward();
    assert.deepStrictEqual(linear.weight.grad.shape, [2, 3]);
    assert.isNull(linear.bias);
  });
});

describe('Loss reduction parameter', () => {
  describe('MSELoss', () => {
    const pred = new Tensor([1.0, 2.0, 3.0]);
    const target = new Tensor([1.5, 2.5, 3.5]);

    it('reduction=mean (default)', () => {
      const loss = new torch.nn.MSELoss();
      const result = loss.forward(pred, target);
      // (0.25 + 0.25 + 0.25) / 3 = 0.25
      assert.closeTo(result.item(), 0.25, 1e-5);
    });

    it('reduction=sum', () => {
      const loss = new torch.nn.MSELoss('sum');
      const result = loss.forward(pred, target);
      // 0.25 + 0.25 + 0.25 = 0.75
      assert.closeTo(result.item(), 0.75, 1e-5);
    });

    it('reduction=none', () => {
      const loss = new torch.nn.MSELoss('none');
      const result = loss.forward(pred, target);
      assert.deepStrictEqual(result.shape, [3]);
      const data = result.toFlatArray();
      assert.closeTo(data[0], 0.25, 1e-5);
      assert.closeTo(data[1], 0.25, 1e-5);
      assert.closeTo(data[2], 0.25, 1e-5);
    });
  });

  describe('L1Loss', () => {
    const pred = new Tensor([1.0, 2.0, 3.0]);
    const target = new Tensor([1.5, 2.5, 3.5]);

    it('reduction=mean (default)', () => {
      const loss = new torch.nn.L1Loss();
      const result = loss.forward(pred, target);
      assert.closeTo(result.item(), 0.5, 1e-5);
    });

    it('reduction=sum', () => {
      const loss = new torch.nn.L1Loss('sum');
      const result = loss.forward(pred, target);
      assert.closeTo(result.item(), 1.5, 1e-5);
    });

    it('reduction=none', () => {
      const loss = new torch.nn.L1Loss('none');
      const result = loss.forward(pred, target);
      assert.deepStrictEqual(result.shape, [3]);
      const data = result.toFlatArray();
      assert.closeTo(data[0], 0.5, 1e-5);
      assert.closeTo(data[1], 0.5, 1e-5);
      assert.closeTo(data[2], 0.5, 1e-5);
    });
  });

  describe('CrossEntropyLoss', () => {
    // Logits for batch of 3, 3 classes
    const input = new Tensor([[2.0, 1.0, 0.1],
                               [0.5, 2.5, 0.3],
                               [0.1, 0.2, 3.0]], { requires_grad: true });
    const target = new Tensor([0, 1, 2]);

    it('reduction=mean (default)', () => {
      const loss = new torch.nn.CrossEntropyLoss();
      const result = loss.forward(input, target);
      assert.deepStrictEqual(result.shape, []);
      assert.closeTo(result.item(), 0.2489, 0.01);
    });

    it('reduction=sum', () => {
      const loss = new torch.nn.CrossEntropyLoss('sum');
      const result = loss.forward(input, target);
      assert.deepStrictEqual(result.shape, []);
      assert.closeTo(result.item(), 0.7467, 0.01);
    });

    it('reduction=none', () => {
      const loss = new torch.nn.CrossEntropyLoss('none');
      const result = loss.forward(input, target);
      assert.deepStrictEqual(result.shape, [3]);
    });

    it('reduction=mean backward', () => {
      const inp = new Tensor([[2.0, 1.0, 0.1],
                               [0.5, 2.5, 0.3],
                               [0.1, 0.2, 3.0]], { requires_grad: true });
      const tgt = new Tensor([0, 1, 2]);
      const loss = new torch.nn.CrossEntropyLoss();
      const result = loss.forward(inp, tgt);
      result.backward();
      assert.deepStrictEqual(inp.grad.shape, [3, 3]);
    });

    it('reduction=sum backward', () => {
      const inp = new Tensor([[2.0, 1.0, 0.1],
                               [0.5, 2.5, 0.3],
                               [0.1, 0.2, 3.0]], { requires_grad: true });
      const tgt = new Tensor([0, 1, 2]);
      const loss = new torch.nn.CrossEntropyLoss('sum');
      const result = loss.forward(inp, tgt);
      result.backward();
      assert.deepStrictEqual(inp.grad.shape, [3, 3]);
    });
  });
});
