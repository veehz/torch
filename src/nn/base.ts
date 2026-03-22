import { Tensor } from '../tensor';
import { Parameter } from './parameter';

export { Parameter } from './parameter';

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

  public named_parameters(prefix: string = ''): [string, Parameter][] {
    const result: [string, Parameter][] = [];
    for (const [name, param] of Object.entries(this._parameters)) {
      const fullName = prefix ? `${prefix}.${name}` : name;
      result.push([fullName, param]);
    }
    for (const [name, module] of Object.entries(this._modules)) {
      const fullName = prefix ? `${prefix}.${name}` : name;
      result.push(...module.named_parameters(fullName));
    }
    return result;
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
