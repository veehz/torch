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
 *   reduction – 'mean' | 'sum' | 'none'
 *
 * Backward:
 *   d_input[i,j] = (softmax(input)[i,j] - 1{j == target[i]}) * scale
 */
class CrossEntropyLossOp extends TorchFunction {
  private N: number = 0;
  private C: number = 0;
  private reduction: string = 'mean';

  protected _forward(input: Tensor, target: Tensor, reduction: string = 'mean'): Tensor {
    this.reduction = reduction;
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
    const perSampleLoss = new Array(N);
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
      perSampleLoss[i] = -logSoftmax;
    }

    let lossData: number[];
    let resultShape: number[];
    if (reduction === 'none') {
      lossData = perSampleLoss;
      resultShape = [N];
    } else if (reduction === 'sum') {
      lossData = [perSampleLoss.reduce((a: number, b: number) => a + b, 0)];
      resultShape = [];
    } else {
      lossData = [perSampleLoss.reduce((a: number, b: number) => a + b, 0) / N];
      resultShape = [];
    }

    const result = new Tensor(lossData, { requires_grad: rg }, { operation: rg ? this : null, shape: resultShape });
    return result;
  }

  protected _backward(dz: Tensor | number): void {
    const [input, target] = this.saved_tensors;
    const [inputFn] = this.next_functions;
    const N = this.N;
    const C = this.C;
    const reduction = this.reduction;

    const inputData = input.data;
    const targetData = target.data;

    let dzData: number[];
    if (typeof dz === 'number') {
      dzData = new Array(reduction === 'none' ? N : 1).fill(dz);
    } else {
      dzData = [...dz.data];
    }

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
      const dzVal = reduction === 'none' ? dzData[i] : dzData[0];
      const scale = reduction === 'mean' ? dzVal / N : dzVal;

      for (let j = 0; j < C; j++) {
        const softmax_j = Math.exp(inputData[rowOffset + j] - maxVal) / sumExp;
        const oneHot = j === t ? 1 : 0;
        grad[rowOffset + j] = (softmax_j - oneHot) * scale;
      }
    }

    const gradTensor = new Tensor(grad, {}, { shape: [N, C] });
    inputFn.backward(gradTensor);
  }
}

registerOperation('cross_entropy_loss', CrossEntropyLossOp);

class LeakyReluOp extends TorchFunction {
  private negative_slope: number;

  protected _forward(input: Tensor, negative_slope: number = 0.01): Tensor {
    this.negative_slope = negative_slope;
    const rg = resultRequiresGrad(input);
    if (rg) {
      this.saved_tensors = [input];
    }
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);

    const inputData = input.data;
    const outputData = inputData.map(v => v > 0 ? v : negative_slope * v);

    return new Tensor(outputData, { requires_grad: rg }, { operation: rg ? this : null, shape: [...input.shape] });
  }

  protected _backward(dz: Tensor): void {
    const [input] = this.saved_tensors;
    const [inputFn] = this.next_functions;
    const ns = this.negative_slope;

    const inputData = input.data;
    const dzData = dz.data;
    const gradData = inputData.map((v, i) => v > 0 ? dzData[i] : ns * dzData[i]);

    inputFn.backward(new Tensor(gradData, {}, { shape: [...input.shape] }));
  }
}
registerOperation('leaky_relu', LeakyReluOp);

class MaxPool2dOp extends TorchFunction {
  private pool_h: number;
  private pool_w: number;
  private stride_h: number;
  private stride_w: number;
  private pad_h: number;
  private pad_w: number;
  private argmax_data: number[];
  private input_ndim: number;

  protected _forward(
    input: Tensor,
    kernel_size: number | number[],
    stride?: number | number[],
    padding: number | number[] = 0
  ): Tensor {
    const kArr = typeof kernel_size === 'number' ? [kernel_size, kernel_size] : kernel_size as number[];
    const sArr = stride === undefined ? kArr : (typeof stride === 'number' ? [stride, stride] : stride as number[]);
    const pArr = typeof padding === 'number' ? [padding, padding] : padding as number[];

    this.pool_h = kArr[0];
    this.pool_w = kArr[1];
    this.stride_h = sArr[0];
    this.stride_w = sArr[1];
    this.pad_h = pArr[0];
    this.pad_w = pArr[1];
    this.input_ndim = input.shape.length;

    const is4d = input.shape.length === 4;
    let N: number, C: number, H: number, W: number;
    if (is4d) {
      [N, C, H, W] = input.shape;
    } else {
      N = 1;
      [C, H, W] = input.shape;
    }

    const H_out = Math.floor((H + 2 * this.pad_h - this.pool_h) / this.stride_h) + 1;
    const W_out = Math.floor((W + 2 * this.pad_w - this.pool_w) / this.stride_w) + 1;

    const rg = resultRequiresGrad(input);
    if (rg) {
      this.saved_tensors = [input];
    }
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);

    const inputData = input.data;
    const totalOut = N * C * H_out * W_out;
    const outputData = new Array(totalOut);
    const argmax = new Array(totalOut);

    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let oh = 0; oh < H_out; oh++) {
          for (let ow = 0; ow < W_out; ow++) {
            let maxVal = -Infinity;
            let maxIdx = -1;
            for (let kh = 0; kh < this.pool_h; kh++) {
              for (let kw = 0; kw < this.pool_w; kw++) {
                const ih = oh * this.stride_h - this.pad_h + kh;
                const iw = ow * this.stride_w - this.pad_w + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  const inputIdx = ((n * C + c) * H + ih) * W + iw;
                  if (inputData[inputIdx] > maxVal) {
                    maxVal = inputData[inputIdx];
                    maxIdx = inputIdx;
                  }
                }
              }
            }
            const outIdx = ((n * C + c) * H_out + oh) * W_out + ow;
            outputData[outIdx] = maxVal;
            argmax[outIdx] = maxIdx;
          }
        }
      }
    }

    this.argmax_data = argmax;
    const outShape = is4d ? [N, C, H_out, W_out] : [C, H_out, W_out];

    return new Tensor(outputData, { requires_grad: rg }, { operation: rg ? this : null, shape: outShape });
  }

  protected _backward(dz: Tensor): void {
    const [input] = this.saved_tensors;
    const [inputFn] = this.next_functions;

    const gradData = new Array(input.dataLength()).fill(0);
    const dzData = dz.data;

    for (let i = 0; i < dzData.length; i++) {
      if (this.argmax_data[i] >= 0) {
        gradData[this.argmax_data[i]] += dzData[i];
      }
    }

    inputFn.backward(new Tensor(gradData, {}, { shape: [...input.shape] }));
  }
}
registerOperation('max_pool2d', MaxPool2dOp);

class NLLLossOp extends TorchFunction {
  private N: number;
  private C: number;
  private reduction: string;

  protected _forward(input: Tensor, target: Tensor, reduction: string = 'mean'): Tensor {
    this.reduction = reduction;
    const rg = resultRequiresGrad(input);
    if (rg) {
      this.saved_tensors = [input, target];
    }
    this.next_functions.push(input.grad_fn ? input.grad_fn : nullOp);

    const N = input.shape[0];
    const C = input.shape[1];
    this.N = N;
    this.C = C;

    const inputData = input.data;
    const targetData = target.data;

    const perSampleLoss = new Array(N);
    for (let i = 0; i < N; i++) {
      const t = Math.round(targetData[i]);
      perSampleLoss[i] = -inputData[i * C + t];
    }

    let lossData: number[];
    let resultShape: number[];
    if (reduction === 'none') {
      lossData = perSampleLoss;
      resultShape = [N];
    } else if (reduction === 'sum') {
      lossData = [perSampleLoss.reduce((a, b) => a + b, 0)];
      resultShape = [];
    } else {
      lossData = [perSampleLoss.reduce((a, b) => a + b, 0) / N];
      resultShape = [];
    }

    return new Tensor(lossData, { requires_grad: rg }, { operation: rg ? this : null, shape: resultShape });
  }

  protected _backward(dz: Tensor | number): void {
    const [input, target] = this.saved_tensors;
    const [inputFn] = this.next_functions;
    const N = this.N;
    const C = this.C;
    const reduction = this.reduction;

    const targetData = target.data;

    let dzData: number[];
    if (typeof dz === 'number') {
      dzData = new Array(reduction === 'none' ? N : 1).fill(dz);
    } else {
      dzData = [...dz.data];
    }

    const gradData = new Array(N * C).fill(0);
    for (let i = 0; i < N; i++) {
      const t = Math.round(targetData[i]);
      const dzVal = reduction === 'none' ? dzData[i] : dzData[0];
      const scale = reduction === 'mean' ? dzVal / N : dzVal;
      gradData[i * C + t] = -scale;
    }

    inputFn.backward(new Tensor(gradData, {}, { shape: [N, C] }));
  }
}
registerOperation('nll_loss', NLLLossOp);
