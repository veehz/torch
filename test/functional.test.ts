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

  describe("Matrix Multiplication", () => {
    test("should multiply two tensors with shape (1)", () => {
      const t1 = new Tensor([10]);
      const t2 = new Tensor([20]);
      const result = torch.matmul(t1, t2);

      expect(Array.from(result.data)).toEqual([200]);
      expect(result.shape).toEqual([1]);
    });

    test("should multiply two tensors with 1 dim", () => {
      const t1 = new Tensor([1, 2, 3, 4]);
      const t2 = new Tensor([5, 6, 7, 8]);
      const result = torch.matmul(t1, t2);

      expect(Array.from(result.data)).toEqual([70]);
      expect(result.shape).toEqual([1]);
    });

    test("should multiply two tensors with shape (2, 3) and (3, 3) with the correct values", () => {
      const t1 = new Tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]);

      const t2 = new Tensor([
        [9, 9, 1],
        [6, 4, 3],
        [5, 5, 6]
      ])

      const result = torch.matmul(t1, t2);

      const expected = [
        36, 32, 25,
        96, 86, 55
      ];

      expect(Array.from(result.data)).toEqual(expected);
      expect(result.shape).toEqual([2, 3]);
    });

    test("should output correct shape", () => {
      function shape_of(shape1: number[], shape2: number[]): number[] {
        const tensor1 = torch.randn(shape1);
        const tensor2 = torch.randn(shape2);
        const result = torch.matmul(tensor1, tensor2);
        return result.shape;
      }

      expect(shape_of([3, 4], [4, 5])).toEqual([3, 5]);
      expect(shape_of([3, 4], [4])).toEqual([3]);
      expect(shape_of([10, 3, 4], [4])).toEqual([10, 3]);
      expect(shape_of([10, 3, 4], [10, 4, 5])).toEqual([10, 3, 5]);
      expect(shape_of([10, 3, 4], [4, 5])).toEqual([10, 3, 5]);
    });
  });

  describe("Transpose", () => {
    test("should transpose a tensor", () => {
      const t = new Tensor([[1, 2], [3, 4]]);
      const result = t.transpose(0, 1);
      expect(Array.from(result.data)).toEqual([1, 3, 2, 4]);
      expect(result.shape).toEqual([2, 2]);
    });
  });
});
