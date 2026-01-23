import { assert } from 'chai';
import { Tensor, add, __left_index__, __right_index__ } from 'torch';

describe('Broadcast Index', () => {
    it('Broadcast indices should be correct in 2d array', () => {
        const arr = new Tensor([
            [0, 0],
            [0, 0],
            [0, 0]
        ]);

        assert.deepStrictEqual(__left_index__(arr, arr).toArray(), [0, 1, 2, 3, 4, 5]);
        assert.deepStrictEqual(__right_index__(arr, arr).toArray(), [0, 1, 2, 3, 4, 5]);
    });

    it('Should get correct value of left and right index', () => {
        const arr = new Tensor([
            [1, 2],
            [3, 4],
            [5, 6]
        ]);

        const zeros = new Tensor([
            [0, 0],
            [0, 0],
            [0, 0]
        ])

        assert.deepStrictEqual(add(arr, zeros).toArray(), [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual(add(zeros, arr).toArray(), [1, 2, 3, 4, 5, 6]);
    });
});