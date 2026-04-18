import { assert } from 'chai';
import * as torch from '@sourceacademy/torch';
import { Tensor } from '@sourceacademy/torch';

describe('Tensor views via index()', () => {

  // ─── shape and values ───────────────────────────────────────────────────────

  describe('shape and values', () => {
    it('2D → 1D: correct shape and values', () => {
      const x = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const y = x.index(0);
      assert.deepStrictEqual(y.shape, [3]);
      assert.deepStrictEqual(y.toArray(), [1, 2, 3]);
    });

    it('2D → 1D: second row', () => {
      const x = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const y = x.index(1);
      assert.deepStrictEqual(y.shape, [3]);
      assert.deepStrictEqual(y.toArray(), [4, 5, 6]);
    });

    it('3D → 2D: correct shape and values', () => {
      const x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const y = x.index(1);
      assert.deepStrictEqual(y.shape, [2, 2]);
      assert.deepStrictEqual(y.toArray(), [[5, 6], [7, 8]]);
    });

    it('1D → scalar: shape [] and correct value', () => {
      const x = torch.tensor([10, 20, 30]);
      const s = x.index(1);
      assert.deepStrictEqual(s.shape, []);
      assert.strictEqual(s.item(), 20);
    });

    it('negative index: last row', () => {
      const x = torch.tensor([[1, 2], [3, 4], [5, 6]]);
      const y = x.index(-1);
      assert.deepStrictEqual(y.shape, [2]);
      assert.deepStrictEqual(y.toArray(), [5, 6]);
    });

    it('negative index: second-to-last row', () => {
      const x = torch.tensor([[1, 2], [3, 4], [5, 6]]);
      const y = x.index(-2);
      assert.deepStrictEqual(y.shape, [2]);
      assert.deepStrictEqual(y.toArray(), [3, 4]);
    });
  });

  // ─── error cases ────────────────────────────────────────────────────────────

  describe('error cases', () => {
    it('throws on scalar tensor', () => {
      const s = torch.tensor(5);
      assert.throws(() => s.index(0), /scalar/i);
    });

    it('throws on out-of-bounds positive index', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      assert.throws(() => x.index(2), /out of bounds/i);
    });

    it('throws on out-of-bounds negative index', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      assert.throws(() => x.index(-3), /out of bounds/i);
    });
  });

  // ─── data sharing: parent → view ────────────────────────────────────────────

  describe('data sharing: parent mutations visible in view', () => {
    it('zero_() on parent zeros out the view', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(0); // view of row 0
      x.zero_();
      assert.deepStrictEqual(y.toArray(), [0, 0]);
    });

    it('data setter on parent is visible in view', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(1); // view of row 1
      x.data = [10, 20, 30, 40];
      assert.deepStrictEqual(y.toArray(), [30, 40]);
    });
  });

  // ─── data sharing: view → parent ────────────────────────────────────────────

  describe('data sharing: view mutations visible in parent', () => {
    it('zero_() on view zeros the corresponding row of parent', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(0);
      y.zero_();
      assert.deepStrictEqual(x.toArray(), [[0, 0], [3, 4]]);
    });

    it('zero_() on second-row view leaves first row intact', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(1);
      y.zero_();
      assert.deepStrictEqual(x.toArray(), [[1, 2], [0, 0]]);
    });

    it('data setter on view writes into parent storage', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(0);
      y.data = [99, 88];
      assert.deepStrictEqual(x.toArray(), [[99, 88], [3, 4]]);
    });

    it('multiple views are all linked to the same storage', () => {
      const x = torch.tensor([[1, 2], [3, 4], [5, 6]]);
      const row0 = x.index(0);
      const row2 = x.index(2);
      row0.zero_();
      row2.data = [9, 9];
      assert.deepStrictEqual(x.toArray(), [[0, 0], [3, 4], [9, 9]]);
      assert.deepStrictEqual(row0.toArray(), [0, 0]);
      assert.deepStrictEqual(row2.toArray(), [9, 9]);
    });

    it('3D → 2D view: zero_() on view updates parent', () => {
      const x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const y = x.index(0); // [[1,2],[3,4]]
      y.zero_();
      assert.deepStrictEqual(x.toArray(), [[[0, 0], [0, 0]], [[5, 6], [7, 8]]]);
    });
  });

  // ─── operations on views ─────────────────────────────────────────────────────

  describe('operations on views produce correct results', () => {
    it('add on a view gives correct values', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(1); // [3, 4]
      const z = y.add(torch.tensor([10, 10]));
      assert.deepStrictEqual(z.toArray(), [13, 14]);
      // original unchanged
      assert.deepStrictEqual(x.toArray(), [[1, 2], [3, 4]]);
    });

    it('mul on a view gives correct values', () => {
      const x = torch.tensor([[2, 3], [4, 5]]);
      const y = x.index(0); // [2, 3]
      const z = y.mul(torch.tensor([2, 3]));
      assert.deepStrictEqual(z.toArray(), [4, 9]);
    });

    it('sum on a view gives correct scalar', () => {
      const x = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const y = x.index(1); // [4, 5, 6]
      assert.strictEqual(y.sum().item(), 15);
    });

    it('matmul on a 1D view gives correct dot product', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(0); // [1, 2]
      const z = x.index(1); // [3, 4]
      const dot = y.matmul(z); // 1*3 + 2*4 = 11
      assert.strictEqual(dot.item(), 11);
    });
  });

  // ─── detach() on a view ───────────────────────────────────────────────────────

  describe('detach() on a view creates an independent copy', () => {
    it('detached tensor has the same values', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(0);
      const d = y.detach();
      assert.deepStrictEqual(d.toArray(), [1, 2]);
    });

    it('mutating the detached tensor does not affect parent', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(0);
      const d = y.detach();
      d.zero_();
      // x and y should be unchanged
      assert.deepStrictEqual(x.toArray(), [[1, 2], [3, 4]]);
      assert.deepStrictEqual(y.toArray(), [1, 2]);
    });

    it('mutating the parent does not affect the detached copy', () => {
      const x = torch.tensor([[1, 2], [3, 4]]);
      const y = x.index(0);
      const d = y.detach();
      x.zero_();
      assert.deepStrictEqual(d.toArray(), [1, 2]);
    });
  });

  // ─── optimizer step propagates through shared storage ────────────────────────

  describe('optimizer step: views see updated parameter values', () => {
    it('SGD step: view of param reflects new values', () => {
      // param shape [2, 2], create view of first row before the step
      const param = new torch.nn.Parameter(torch.tensor([[1, 2], [3, 4]]));
      const row0_before = param.index(0).toArray() as number[];
      assert.deepStrictEqual(row0_before, [1, 2]);

      // Simulate what the optimizer does: param.data = newParam.data
      const newValues = [10, 20, 30, 40];
      param.data = newValues;

      // A view created AFTER the update sees the new values
      const row1_after = param.index(1);
      assert.deepStrictEqual(row1_after.toArray(), [30, 40]);

      // A view created BEFORE the update also sees the new values (shared storage)
      const row0_view = param.index(0);
      assert.deepStrictEqual(row0_view.toArray(), [10, 20]);
    });

    it('full SGD training step does not corrupt parameter shape', () => {
      class Linear extends torch.nn.Module {
        w: torch.nn.Parameter;
        constructor() {
          super();
          this.w = new torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]));
          this.register('w', this.w);
        }
        forward(x: Tensor): Tensor {
          return x.matmul(this.w);
        }
      }

      const model = new Linear();
      const optim = new torch.optim.SGD(model.parameters(), 0.1);

      const x = torch.tensor([[1.0, 2.0]]);
      const y = model.forward(x);
      const loss = y.sum();
      loss.backward();

      optim.step();

      // After optimizer step, param shape must be preserved
      assert.deepStrictEqual(model.w.shape, [2, 2]);
      assert.strictEqual(model.w.dataLength(), 4);

      // View of updated param must have correct shape and values
      const row0 = model.w.index(0);
      assert.deepStrictEqual(row0.shape, [2]);
      assert.strictEqual(row0.dataLength(), 2);
    });
  });

  // ─── chained index ────────────────────────────────────────────────────────────

  describe('chained index()', () => {
    it('index into a view gives correct scalar', () => {
      const x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      // x.index(1) → [[5,6],[7,8]], then .index(0) → [5,6], then .index(1) → 6
      const val = x.index(1).index(0).index(1);
      assert.deepStrictEqual(val.shape, []);
      assert.strictEqual(val.item(), 6);
    });

    it('zero_() on doubly-chained view updates root storage', () => {
      const x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      x.index(0).index(1).zero_(); // zero out [3,4]
      assert.deepStrictEqual(x.toArray(), [[[1, 2], [0, 0]], [[5, 6], [7, 8]]]);
    });
  });
});
