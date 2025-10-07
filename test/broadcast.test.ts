import { _broadcast_shape } from '../src/broadcasting';

describe('Broadcasting', () => {
    test('Broadcast shape of same dimensions', () => {
        expect(_broadcast_shape([1, 3, 1], [4, 3, 6])).toEqual([4, 3, 6]);
        expect(_broadcast_shape([4, 3, 6], [1, 3, 1])).toEqual([4, 3, 6]);
        expect(_broadcast_shape([2, 2, 2], [2, 2, 2])).toEqual([2, 2, 2]);

        expect(() => _broadcast_shape([1, 2, 1], [1, 3, 1])).toThrow(Error);
    });

    test('Broadcast shape of different dimensions', () => {
        expect(_broadcast_shape([2, 3, 1], [3, 1])).toEqual([2, 3, 1]);
        expect(_broadcast_shape([3, 1], [2, 3, 1])).toEqual([2, 3, 1]);
        expect(_broadcast_shape([1], [2, 2, 2])).toEqual([2, 2, 2]);
        expect(_broadcast_shape([2], [])).toEqual([2]);
        expect(_broadcast_shape([1, 4], [3, 3, 1])).toEqual([3, 3, 4]);

        expect(() => _broadcast_shape([1, 2], [3, 3, 6])).toThrow(Error);
    });
});