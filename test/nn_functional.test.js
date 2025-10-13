import { assert } from 'chai';
import * as torch from 'torch';

describe('NN Functional', () => {
  describe('Relu', () => {
    it('should forward a tensor', () => {
      const x = new torch.Tensor([1, -2, 3, -4, 5]);
      const result = torch.nn.functional.relu(x);
      assert.deepStrictEqual(Array.from(result.toArray()), [1, 0, 3, 0, 5]);
      assert.deepStrictEqual(result.shape, [5]);
    });
  });
});
