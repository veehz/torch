
import * as torch from 'torch';
import { Tensor } from 'torch';
import { assert } from 'chai';

describe('BCELoss', () => {
    it('scalar', () => {
        const input = new Tensor(0.123, { requires_grad: true });
        const target = new Tensor(0.9876);

        const loss = new torch.nn.BCELoss();
        const result = loss.forward(input, target);

        assert.closeTo(result.item(), 2.0712, 0.001);
    })

    it('should calculate binary cross entropy loss correctly', () => {
        /* Python:
         import torch
         x, y
         */
        const m = new torch.nn.Sigmoid()
        const input = new Tensor([[0.5424, 1.3919],
                                [-1.0297, -0.6352],
                                [0.5700, -1.0037]],
                                { requires_grad: true });
        const target = new Tensor([[0.8340, 0.4923],
                                [0.7729, 0.7560],
                                [0.5616, 0.0999]]);

        const loss = new torch.nn.BCELoss();
        const result = loss.forward(m.forward(input), target);

        assert.closeTo(result.item(), 0.7657, 0.001);
    });
});

describe('NLLLoss', () => {
  it('basic mean reduction', () => {
    const loss_fn = new torch.nn.NLLLoss();
    const logProbs = torch.tensor([
      [Math.log(0.1), Math.log(0.3), Math.log(0.6)],
      [Math.log(0.7), Math.log(0.2), Math.log(0.1)],
    ]);
    const target = torch.tensor([2, 0]);
    const out = loss_fn.forward(logProbs, target);
    const expected = -(Math.log(0.6) + Math.log(0.7)) / 2;
    assert.closeTo(out.item(), expected, 1e-5);
  });

  it('gradient flows back to input', () => {
    const loss_fn = new torch.nn.NLLLoss();
    const logProbs = torch.tensor([[-1.0, -2.0, -0.5], [-0.8, -1.2, -0.3]], true);
    const target = torch.tensor([0, 2]);
    loss_fn.forward(logProbs, target).backward();
    assert.closeTo(logProbs.grad.toArray()[0][0], -0.5, 1e-6);
    assert.closeTo(logProbs.grad.toArray()[1][2], -0.5, 1e-6);
    assert.closeTo(logProbs.grad.toArray()[0][1],  0.0, 1e-6);
  });

  it('reduction=sum', () => {
    const loss_fn = new torch.nn.NLLLoss('sum');
    const logProbs = torch.tensor([[-1.0, -0.5], [-0.8, -0.3]]);
    const target = torch.tensor([1, 0]);
    assert.closeTo(loss_fn.forward(logProbs, target).item(), 1.3, 1e-5);
  });

  it('reduction=none returns per-sample losses', () => {
    const loss_fn = new torch.nn.NLLLoss('none');
    const logProbs = torch.tensor([[-1.0, -0.5], [-0.8, -0.3]]);
    const target = torch.tensor([1, 0]);
    const out = loss_fn.forward(logProbs, target);
    assert.deepStrictEqual(out.shape, [2]);
    assert.closeTo(out.toArray()[0], 0.5, 1e-5);
    assert.closeTo(out.toArray()[1], 0.8, 1e-5);
  });
});
