import { UnaryFunctionMixin } from '../functions/mixin';
import { TorchFunction, resultRequiresGrad, nullOp } from '../functions/base';
import { registerOperation } from '../functions/registry';
import { Tensor } from '../tensor';

const Relu = UnaryFunctionMixin(
  (a: number[], x: number) => Math.max(a[x], 0),
  (a, aFn, dz) => {
    aFn.backward(dz.mul(a.gt(0)));
  },
  "relu"
);

const Sigmoid = UnaryFunctionMixin(
  (a: number[], x: number) => 1 / (1 + Math.exp(-a[x])),
  (a, aFn, dz) => {
    const res = a.sigmoid();
    aFn.backward(res.mul(res.mul(-1).add(1)).mul(dz));
  },
  "sigmoid"
);

/**
 * CrossEntropyLoss operation.
 *
 * Forward:
 *   input  – (N, C) logits (unnormalized scores)
 *   target – (N,)   integer class indices in [0, C)
 *   Returns a scalar loss tensor.
 *
 * Backward:
 *   d_input[i,j] = (softmax(input)[i,j] - 1{j == target[i]}) / N
 */
class CrossEntropyLossOp extends TorchFunction {
  private N: number = 0;
  private C: number = 0;

  protected _forward(input: Tensor, target: Tensor): Tensor {
    const rg = resultRequiresGrad(input);
    if (rg) {
      this.saved_tensors = [input, target];
    }
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);

    const shape = input.shape; // (N, C)
    const N = shape[0];
    const C = shape[1];
    this.N = N;
    this.C = C;

    const inputData = input.data;
    const targetData = target.data;

    // Numerically stable log-softmax + gather
    let totalLoss = 0;
    for (let i = 0; i < N; i++) {
      const rowOffset = i * C;

      // Find max for numerical stability
      let maxVal = -Infinity;
      for (let j = 0; j < C; j++) {
        if (inputData[rowOffset + j] > maxVal) {
          maxVal = inputData[rowOffset + j];
        }
      }

      // Compute log(sum(exp(x - max)))
      let sumExp = 0;
      for (let j = 0; j < C; j++) {
        sumExp += Math.exp(inputData[rowOffset + j] - maxVal);
      }
      const logSumExp = Math.log(sumExp);

      // log_softmax for the target class
      const t = targetData[i];
      const logSoftmax = inputData[rowOffset + t] - maxVal - logSumExp;
      totalLoss -= logSoftmax;
    }

    const loss = totalLoss / N;
    const result = new Tensor([loss], { requires_grad: rg }, { operation: rg ? this : null, shape: [] });
    return result;
  }

  protected _backward(dz: Tensor | number): void {
    const [input, target] = this.saved_tensors;
    const [inputFn] = this.next_functions;
    const N = this.N;
    const C = this.C;

    const inputData = input.data;
    const targetData = target.data;
    const dzVal = typeof dz === 'number' ? dz : dz.data[0];

    // Gradient: (softmax - one_hot) / N * dz
    const grad = new Array(N * C);
    for (let i = 0; i < N; i++) {
      const rowOffset = i * C;

      // Compute softmax for this row
      let maxVal = -Infinity;
      for (let j = 0; j < C; j++) {
        if (inputData[rowOffset + j] > maxVal) {
          maxVal = inputData[rowOffset + j];
        }
      }
      let sumExp = 0;
      for (let j = 0; j < C; j++) {
        sumExp += Math.exp(inputData[rowOffset + j] - maxVal);
      }

      const t = targetData[i];
      for (let j = 0; j < C; j++) {
        const softmax_j = Math.exp(inputData[rowOffset + j] - maxVal) / sumExp;
        const oneHot = j === t ? 1 : 0;
        grad[rowOffset + j] = (softmax_j - oneHot) / N * dzVal;
      }
    }

    const gradTensor = new Tensor(grad, {}, { shape: [N, C] });
    inputFn.backward(gradTensor);
  }
}

registerOperation('cross_entropy_loss', CrossEntropyLossOp);
