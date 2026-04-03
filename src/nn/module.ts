import { Module, Parameter } from "./base";
import { rand } from "../creation";
import * as functional from "./functional";
import { softmax as _softmax } from "../functions/functional";
import { Tensor } from "../tensor";

export class Linear extends Module {
  public weight: Parameter;
  public bias: Parameter | null;

  constructor(in_features: number, out_features: number, bias: boolean = true) {
    super();
    const k = Math.sqrt(1 / in_features);

    this.weight = new Parameter(
      rand([out_features, in_features])
        .mul(2 * k)
        .sub(k)
    );
    this.register('weight', this.weight);

    if (bias) {
      this.bias = new Parameter(
        rand([out_features])
          .mul(2 * k)
          .sub(k)
      );
      this.register('bias', this.bias);
    } else {
      this.bias = null;
    }
  }

  forward(input: Tensor) {
    const out = input.matmul(this.weight.transpose(0, 1));
    return this.bias ? out.add(this.bias) : out;
  }
}

export class ReLU extends Module {
  constructor() {
    super();
  }

  forward(input: Tensor) {
    return functional.relu(input);
  }
}

export class Sigmoid extends Module {
  constructor() {
    super();
  }

  forward(input: Tensor) {
    return functional.sigmoid(input);
  }
}

abstract class _ConvNd extends Module {
  public weight: Parameter;
  public bias: Parameter | null;

  public in_channels: number;
  public out_channels: number;
  public kernel_size: number | number[];
  public stride: number | number[];
  public padding: number | number[];
  public dilation: number | number[];
  public groups: number;

  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | number[],
    stride: number | number[],
    padding: number | number[],
    dilation: number | number[],
    groups: number,
    bias: boolean,
    dims: number
  ) {
    super();

    this.in_channels = in_channels;
    this.out_channels = out_channels;
    this.kernel_size = kernel_size;
    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;

    if (in_channels % groups !== 0) {
      throw new Error('in_channels must be divisible by groups');
    }
    if (out_channels % groups !== 0) {
      throw new Error('out_channels must be divisible by groups');
    }

    const kernel_arr = typeof kernel_size === 'number' ? new Array(dims).fill(kernel_size) : kernel_size;
    const kernel_vol = kernel_arr.reduce((a: number, b: number) => a * b, 1);

    const k = Math.sqrt(groups / (in_channels * kernel_vol));

    this.weight = new Parameter(
      rand([out_channels, in_channels / groups, ...kernel_arr])
        .mul(2 * k)
        .sub(k)
    );

    this.register('weight', this.weight);

    if (bias) {
      this.bias = new Parameter(
        rand([out_channels])
          .mul(2 * k)
          .sub(k)
      );
      this.register('bias', this.bias);
    } else {
      this.bias = null;
    }
  }

  abstract forward(input: Tensor): Tensor;
}

export class Conv1d extends _ConvNd {
  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | number[],
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    groups: number = 1,
    bias: boolean = true
  ) {
    super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 1);
  }

  forward(input: Tensor) {
    return functional.conv1d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}

export class Conv2d extends _ConvNd {
  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | number[],
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    groups: number = 1,
    bias: boolean = true
  ) {
    super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 2);
  }

  forward(input: Tensor) {
    return functional.conv2d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}

export class LeakyReLU extends Module {
  private negative_slope: number;

  constructor(negative_slope: number = 0.01) {
    super();
    this.negative_slope = negative_slope;
  }

  forward(input: Tensor) {
    return functional.leaky_relu(input, this.negative_slope);
  }
}

export class MaxPool2d extends Module {
  private kernel_size: number | number[];
  private stride: number | number[] | undefined;
  private padding: number | number[];

  constructor(
    kernel_size: number | number[],
    stride?: number | number[],
    padding: number | number[] = 0
  ) {
    super();
    this.kernel_size = kernel_size;
    this.stride = stride;
    this.padding = padding;
  }

  forward(input: Tensor) {
    return functional.max_pool2d(input, this.kernel_size, this.stride, this.padding);
  }
}

export class Dropout extends Module {
  private p: number;

  constructor(p: number = 0.5) {
    super();
    this.p = p;
  }

  forward(input: Tensor) {
    if (!this.training || this.p === 0) {
      return input;
    }
    if (this.p === 1) {
      return input.mul(0);
    }
    const scale = 1 / (1 - this.p);
    const maskData = input.toFlatArray().map(() => (Math.random() > this.p ? scale : 0));
    const mask = new Tensor(maskData, {}, { shape: [...input.shape] });
    return input.mul(mask);
  }
}

export class Softmax extends Module {
  private dim: number;

  constructor(dim: number) {
    super();
    this.dim = dim;
  }

  forward(input: Tensor) {
    return _softmax(input, this.dim);
  }
}

export class Flatten extends Module {
  private start_dim: number;
  private end_dim: number;

  constructor(start_dim: number = 1, end_dim: number = -1) {
    super();
    this.start_dim = start_dim;
    this.end_dim = end_dim;
  }

  forward(input: Tensor) {
    return input.flatten(this.start_dim, this.end_dim);
  }
}

export class Conv3d extends _ConvNd {
  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | number[],
    stride: number | number[] = 1,
    padding: number | number[] = 0,
    dilation: number | number[] = 1,
    groups: number = 1,
    bias: boolean = true
  ) {
    super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 3);
  }

  forward(input: Tensor) {
    return functional.conv3d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
