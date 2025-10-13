import { assert } from 'chai';
import { _broadcast_shape } from '../src/broadcasting';

describe('Broadcasting', () => {
  it('Broadcast shape of same dimensions', () => {
    assert.deepStrictEqual(_broadcast_shape([1, 3, 1], [4, 3, 6]), [4, 3, 6]);
    assert.deepStrictEqual(_broadcast_shape([4, 3, 6], [1, 3, 1]), [4, 3, 6]);
    assert.deepStrictEqual(_broadcast_shape([2, 2, 2], [2, 2, 2]), [2, 2, 2]);

    assert.throws(() => _broadcast_shape([1, 2, 1], [1, 3, 1]), Error);
  });

  it('Broadcast shape of different dimensions', () => {
    assert.deepStrictEqual(_broadcast_shape([2, 3, 1], [3, 1]), [2, 3, 1]);
    assert.deepStrictEqual(_broadcast_shape([3, 1], [2, 3, 1]), [2, 3, 1]);
    assert.deepStrictEqual(_broadcast_shape([1], [2, 2, 2]), [2, 2, 2]);
    assert.deepStrictEqual(_broadcast_shape([2], []), [2]);
    assert.deepStrictEqual(_broadcast_shape([1, 4], [3, 3, 1]), [3, 3, 4]);

    assert.throws(() => _broadcast_shape([1, 2], [3, 3, 6]), Error);
  });
});
