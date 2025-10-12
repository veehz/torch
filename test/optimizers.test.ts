import * as torch from "../src/index";

describe("Optimizers", () => {
  describe("SGD", () => {
    test("should update parameters", () => {
      const x = new torch.Tensor([1.0], { requires_grad: true });
      const y = x.mul(new torch.Tensor(2.0));
      y.backward();

      const sgd = new torch.optim.SGD([x], 0.01);
      sgd.step();
      expect(sgd.params[0].data[0]).toBeCloseTo(0.98);
    });
  });
});