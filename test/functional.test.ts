import * as torch from "../src/index";
import { Tensor } from "../src/index";

describe("Functional", () => {
  describe("Addition", () => {
    test("should add two tensors with same shape", () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.add(t1, t2);

      expect(Array.from(result.data)).toEqual([30]);
      expect(result.shape).toEqual([1]);
    });

    test("should add tensors with different shapes (broadcasting)", () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = torch.add(t1, t2);

      expect(Array.from(result.data)).toEqual([11, 21, 31]);
      expect(result.shape).toEqual([3]);
    });

    test("should add two 1D tensors of same length", () => {
      const t1 = new Tensor([1, 2, 3]);
      const t2 = new Tensor([4, 5, 6]);
      const result = torch.add(t1, t2);

      expect(Array.from(result.data)).toEqual([5, 7, 9]);
      expect(result.shape).toEqual([3]);
    });
  });

  describe("Multiplication", () => {
    test("should multiply two tensors with same shape", () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.mul(t1, t2);

      expect(Array.from(result.data)).toEqual([200]);
      expect(result.shape).toEqual([1]);
    });
  });

  describe("should multiply tensors with different shapes (broadcasting)", () => {
    test("should multiply two tensors with different shapes (broadcasting)", () => {
      const t1 = new Tensor([10, 20, 30]);
      const t2 = new Tensor([1]);
      const result = torch.mul(t1, t2);

      expect(Array.from(result.data)).toEqual([10, 20, 30]);
      expect(result.shape).toEqual([3]);
    });
  });
});
