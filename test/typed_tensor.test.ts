import { assert } from 'chai';
import * as torch from 'torch';
import { Tensor, FloatTensor, LongTensor } from 'torch';

describe('FloatTensor', () => {
  it('is an instance of Tensor', () => {
    assert.instanceOf(new FloatTensor([1.5, 2.5]), Tensor);
  });

  it('preserves float values unchanged', () => {
    const data = new FloatTensor([1.1, 2.9, -3.7]).toArray() as number[];
    assert.closeTo(data[0],  1.1, 1e-9);
    assert.closeTo(data[1],  2.9, 1e-9);
    assert.closeTo(data[2], -3.7, 1e-9);
  });

  it('works with nested (2D) data', () => {
    assert.deepStrictEqual(new FloatTensor([[1.5, 2.5], [3.5, 4.5]]).shape, [2, 2]);
  });

  it('accepts requires_grad option', () => {
    assert.isTrue(new FloatTensor([1.0, 2.0], { requires_grad: true }).requires_grad);
  });

  it('participates in autograd', () => {
    const t = new FloatTensor([2.0, 3.0], { requires_grad: true });
    t.sum().backward();
    assert.deepStrictEqual((t.grad!.toArray() as number[]), [1, 1]);
  });

  it('also accessible as torch.FloatTensor', () => {
    assert.instanceOf(new torch.FloatTensor([1.5, 2.5]), Tensor);
  });
});

describe('LongTensor', () => {
  it('is an instance of Tensor', () => {
    assert.instanceOf(new LongTensor([1, 2, 3]), Tensor);
  });

  it('truncates positive floats toward zero', () => {
    assert.deepStrictEqual(new LongTensor([1.1, 1.9, 2.0]).toArray(), [1, 1, 2]);
  });

  it('truncates negative floats toward zero (not floor)', () => {
    assert.deepStrictEqual(new LongTensor([-1.1, -1.9, -2.0]).toArray(), [-1, -1, -2]);
  });

  it('works with nested (2D) data', () => {
    const t = new LongTensor([[1.7, 2.3], [3.9, -4.1]]);
    assert.deepStrictEqual(t.shape, [2, 2]);
    assert.deepStrictEqual(t.toArray(), [[1, 2], [3, -4]]);
  });

  it('accepts integer data unchanged', () => {
    assert.deepStrictEqual(new LongTensor([0, 1, 2, 3]).toArray(), [0, 1, 2, 3]);
  });

  it('accepts requires_grad option and truncates', () => {
    const t = new LongTensor([1.5, 2.5], { requires_grad: true });
    assert.isTrue(t.requires_grad);
    assert.deepStrictEqual(t.toArray(), [1, 2]);
  });

  it('also accessible as torch.LongTensor', () => {
    const t = new torch.LongTensor([1.9, -1.9]);
    assert.instanceOf(t, Tensor);
    assert.deepStrictEqual(t.toArray(), [1, -1]);
  });
});
