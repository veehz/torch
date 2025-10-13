import * as torch from "../src/index";

describe("NN Functional", () => {
  describe("Relu", () => {
    test("should forward a tensor", () => {
      const x = new torch.Tensor([1, -2, 3, -4, 5]);
      const result = torch.nn.functional.relu(x);
      expect(Array.from(result.toArray())).toEqual([1, 0, 3, 0, 5]);
      expect(result.shape).toEqual([5]);
    });
  });
});