import { assert } from 'chai';
import { Tensor } from 'torch';

describe('Tensor', () => {
  describe('Constructor', () => {
    it('should create a tensor with correct data and shape', () => {
      const tensor = new Tensor([10, 20, 30]);
      assert.deepStrictEqual(tensor.data, [10, 20, 30]);
      assert.deepStrictEqual(tensor.shape, [3]);
    });
  });
});