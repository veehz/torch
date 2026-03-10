import { Tensor } from '../tensor';
import { NestedNumberArray } from '../tensor';
import { TorchFunction } from '../functions/base';

export class Parameter extends Tensor {
  constructor(
    data: NestedNumberArray | Tensor | Parameter,
    // Default to requires_grad=true
    options: { requires_grad?: boolean } = {
      requires_grad: true
    },
    internal_options: { operation?: TorchFunction; shape?: number[] } = {}
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

export class Sequential extends Module {
  private _modulesArr: Module[];

  constructor(...modules: Module[]) {
    super();
    this._modulesArr = modules;
    for (let i = 0; i < modules.length; i++) {
      this.register(i.toString(), modules[i]);
    }
  }

  append(module: Module): this {
    this.register(this._modulesArr.length.toString(), module);
    this._modulesArr.push(module);
    return this;
  }

  extend(sequential: Sequential): this {
    for (const module of sequential._modulesArr) {
      this.append(module);
    }
    return this;
  }

  insert(index: number, module: Module): this {
    this._modulesArr.splice(index, 0, module);
    for (let i = index; i < this._modulesArr.length; i++) {
      this.register(i.toString(), this._modulesArr[i]);
    }
    return this;
  }

  forward(input: Tensor) {
    let x = input;
    for (const module of this._modulesArr) {
      x = module.forward(x);
    }
    return x;
  }
}
