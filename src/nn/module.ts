import { Tensor } from "../tensor";
import { NestedNumberArray } from "../tensor";
import { Operation } from "../operations/base";
import { rand } from "../creation";
import { functional } from ".";

export class Parameter extends Tensor {
  constructor(
    data: NestedNumberArray | Tensor | Parameter,
    // Default to requires_grad=true
    options: { requires_grad?: boolean } = {
        requires_grad: true,
    },
    internal_options: { operation?: Operation; shape?: number[] } = {}
  ) {
    if (data instanceof Tensor) {
      super(data.data, { requires_grad: true }, { shape: data.shape });
    } else if (data instanceof Parameter) {
      super(data.data, { requires_grad: true }, { shape: data.shape });
    } else {
      super(data, options, internal_options);
    }
  }
}

export abstract class Module {
  private _modules: { [key: string]: Module };
  private _parameters: { [key: string]: Parameter };

  constructor() {
    this._parameters = {};
    this._modules = {};
  }

  private register_parameter(parameter_name: string, parameter: Parameter) {
    this._parameters[parameter_name] = parameter;
  }

  private register_module(module_name: string, module: Module) {
    this._modules[module_name] = module;
  }

  protected register(name: string, value: Parameter | Module) {
    if (value instanceof Parameter) {
      this.register_parameter(name, value);
    } else {
      this.register_module(name, value);
    }
  }

  public abstract forward(...args: Tensor[]): Tensor;

  public parameters(): Parameter[] {
    let params: Parameter[] = Object.values(this._parameters);
    for (const module of Object.values(this._modules)) {
      params = params.concat(module.parameters());
    }
    return params;
  }
}

export class Linear extends Module {
  private weight: Parameter;
  private bias: Parameter;

  constructor(in_features: number, out_features: number) {
    super();
    const k = Math.sqrt(1 / in_features);

    this.weight = new Parameter(rand([out_features, in_features]).mul(2 * k).sub(k));
    this.bias = new Parameter(rand([out_features]).mul(2 * k).sub(k));

    this.register("weight", this.weight);
    this.register("bias", this.bias);
  }

  forward(input: Tensor) {
    return input.matmul(this.weight.transpose(0, 1)).add(this.bias);
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
