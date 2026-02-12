
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
