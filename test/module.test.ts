import * as torch from "../src/index";

describe("Module", () => {
  describe("Linear", () => {
    test("should create a linear module", () => {
      const linear = new torch.nn.Linear(10, 20);
      expect(linear.parameters().length).toEqual(2);
      expect(linear.parameters()[0].shape).toEqual([20, 10]);
      expect(linear.parameters()[1].shape).toEqual([20]);
    });

    test("should forward a tensor", () => {
      const linear = new torch.nn.Linear(10, 20);
      const input = new torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const output = linear.forward(input);
      expect(output.shape).toEqual([20]);
    });

    test("should forward a tensor (2D)", () => {
      // example from https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
      const m = new torch.nn.Linear(20, 30);
      const input = torch.randn(128, 20);
      const output = m.forward(input);
      expect(output.shape).toEqual([128, 30]);
    });
  });
});