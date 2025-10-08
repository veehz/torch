import gpu from '../gpu';
import { _broadcast_shape, _get_original_index_kernel, _pad_shape } from '../broadcasting';
import { Tensor } from '../tensor';
// import * as utils from "../utils";
import { Operation, BinaryOperation, UnaryOperation } from './base';

/* This did not work as it doesn't support dynamic function registration */
// export function _broadcast_operation(
//   f: (a: number, b: number) => number
// ): (a: Tensor, b: Tensor) => Tensor {
//   return (a: Tensor, b: Tensor) => {
//     const broadcast_shape = _broadcast_shape(a.shape, b.shape);

//     gpu.addFunction(f, {
//       returnType: 'Number',
//       argumentTypes: {
//         a: 'Number',
//         b: 'Number'
//       }
//     });

//     const kernel = gpu
//       .createKernel(function (
//         a: number[],
//         as: number[],
//         b: number[],
//         bs: number[],
//         bcs: number[]
//       ) {
//         const a_index = _get_original_index_gpu(as, bcs, this.thread.x);
//         const b_index = _get_original_index_gpu(bs, bcs, this.thread.x);

//         return f(a[a_index], b[b_index]);
//       })
//       .setConstants({ shape_length: broadcast_shape.length })
//       .setOutput([broadcast_shape.reduce((acc, val) => acc * val, 1)])
//       .setFunctions([f]);

//     return new Tensor(
//       kernel(a.data, a.shape, b.data, b.shape, broadcast_shape) as number[]
//     );
//   };
// }

export function _add(a: number[], b: number[]): number[] {
  const kernel = gpu
    .createKernel(function (a: number[], b: number[]) {
      return a[this.thread.x] + b[this.thread.x];
    })
    .setOutput([a.length]);

  return kernel(a, b) as number[];
}

export function _add_broadcast(
  a: number[],
  a_shape: number[],
  b: number[],
  b_shape: number[]
): number[] {
  const broadcast_shape = _broadcast_shape(a_shape, b_shape);

  const kernel = gpu.createKernel(
    function (
      a: number[],
      a_shape: number[],
      b: number[],
      b_shape: number[],
      broadcast_shape: number[]
    ) {
      const a_index = _get_original_index_kernel(a_shape, broadcast_shape, this.thread.x);
      const b_index = _get_original_index_kernel(b_shape, broadcast_shape, this.thread.x);

      return a[a_index] + b[b_index];
    },
    {
      constants: {
        shape_length: broadcast_shape.length
      },
      output: [broadcast_shape.reduce((acc, val) => acc * val, 1)]
    }
  );

  return kernel(a, a_shape, b, b_shape, broadcast_shape) as number[];
}

// export function _add_tensor(
//   a: Tensor,
//   b: Tensor,
//   operation: Operation | null = null
// ): Tensor {
//   const a_shape = a.shape;
//   const b_shape = b.shape;

//   const result = _add_broadcast(a.data, a_shape, b.data, b_shape);

//   return new Tensor(
//     result,
//     { requires_grad: a.requires_grad || b.requires_grad },
//     { operation: operation }
//   );
// }

// export function _add_tensor(a: Tensor, b: Tensor): Tensor {
//     function _add_(a: number, b: number): number { return a + b }
//     return _broadcast_operation(_add_)(a, b);
// }

export function _add_tensor(a: Tensor, b: Tensor, operation: Operation | null = null): Tensor {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);

  const kernel = gpu.createKernel(
    function (a: number[], as: number[], b: number[], bs: number[], bcs: number[]) {
      const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
      const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);

      return a[a_index] + b[b_index];
    },
    {
      constants: {
        shape_length: broadcast_shape.length
      },
      output: [broadcast_shape.reduce((acc, val) => acc * val, 1)]
    }
  );

  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape) as number[],
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation: operation, shape: broadcast_shape }
  );
}

export class Add extends BinaryOperation {
  cache: [Tensor, Tensor];
  forward(a: Tensor, b: Tensor): Tensor {
    this.cache = [a, b];
    return _add_tensor(a, b, this);
  }
  backward(dz: Tensor): void {
    const [a, b] = this.cache;
    a.backward(dz);
    b.backward(dz);
  }
}

export function _sub_tensor(a: Tensor, b: Tensor, operation: Operation | null = null): Tensor {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);

  const kernel = gpu.createKernel(
    function (a: number[], as: number[], b: number[], bs: number[], bcs: number[]) {
      const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
      const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);

      return a[a_index] - b[b_index];
    },
    {
      constants: {
        shape_length: broadcast_shape?.length || 0
      },
      output: [broadcast_shape.reduce((acc, val) => acc * val, 1)]
    }
  );

  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape) as number[],
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation: operation, shape: broadcast_shape }
  );
}

export class Sub extends BinaryOperation {
  cache: [Tensor, Tensor];
  forward(a: Tensor, b: Tensor): Tensor {
    this.cache = [a, b];
    return _sub_tensor(a, b, this);
  }
  backward(dz: Tensor): void {
    const [a, b] = this.cache;
    a.backward(dz);
    b.backward(dz.mul(new Tensor(-1)));
  }
}

export function _mul_tensor(a: Tensor, b: Tensor, operation: Operation | null = null): Tensor {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = gpu.createKernel(
    function (a: number[], as: number[], b: number[], bs: number[], bcs: number[]) {
      const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
      const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);

      return a[a_index] * b[b_index];
    },
    {
      constants: {
        shape_length: broadcast_shape.length
      },
      output: [broadcast_shape.reduce((acc, val) => acc * val, 1)]
    }
  );

  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape) as number[],
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation: operation, shape: broadcast_shape }
  );
}

export class Mul extends BinaryOperation {
  cache: [Tensor, Tensor];
  forward(a: Tensor, b: Tensor): Tensor {
    this.cache = [a, b];
    return _mul_tensor(a, b, this);
  }
  backward(dz: Tensor): void {
    const [a, b] = this.cache;
    a.backward(dz.mul(b));
    b.backward(dz.mul(a));
  }
}

export function _div_tensor(a: Tensor, b: Tensor, operation: Operation | null = null): Tensor {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = gpu.createKernel(
    function (a: number[], as: number[], b: number[], bs: number[], bcs: number[]) {
      const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
      const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);

      return a[a_index] / b[b_index];
    },
    {
      constants: {
        shape_length: broadcast_shape.length
      },
      output: [broadcast_shape.reduce((acc, val) => acc * val, 1)]
    }
  );

  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape) as number[],
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation: operation, shape: broadcast_shape }
  );
}

export class Div extends BinaryOperation {
  cache: [Tensor, Tensor];
  forward(a: Tensor, b: Tensor): Tensor {
    this.cache = [a, b];
    return _div_tensor(a, b, this);
  }
  backward(dz: Tensor): void {
    const [a, b] = this.cache;
    a.backward(dz.div(b));
    b.backward(dz.mul(a).mul(new Tensor(-1)).div(b).div(b));
  }
}

export class Sum extends UnaryOperation {
  cache: [Tensor];
  forward(a: Tensor): Tensor {
    this.cache = [a];
    return new Tensor(
      a.data.reduce((acc, val) => acc + val, 0),
      { requires_grad: a.requires_grad },
      { operation: this }
    );
  }

  backward(dz: Tensor): void {
    const [a] = this.cache;
    const result = new Tensor(Array(a.data.length).fill(dz.data[0]));
    a.backward(result);
  }
}

export function _pow_tensor(a: Tensor, b: Tensor, operation: Operation | null = null): Tensor {
  const broadcast_shape = _broadcast_shape(a.shape, b.shape);
  const padded_a_shape = _pad_shape(a.shape, broadcast_shape);
  const padded_b_shape = _pad_shape(b.shape, broadcast_shape);
  const kernel = gpu.createKernel(
    function (a: number[], as: number[], b: number[], bs: number[], bcs: number[]) {
      const a_index = _get_original_index_kernel(as, bcs, this.thread.x);
      const b_index = _get_original_index_kernel(bs, bcs, this.thread.x);

      return Math.pow(a[a_index], b[b_index]);
    },
    {
      constants: {
        shape_length: broadcast_shape.length
      },
      output: [broadcast_shape.reduce((acc, val) => acc * val, 1)]
    }
  );

  return new Tensor(
    kernel(a.data, padded_a_shape, b.data, padded_b_shape, broadcast_shape) as number[],
    { requires_grad: a.requires_grad || b.requires_grad },
    { operation: operation, shape: broadcast_shape }
  );
}

export class Pow extends BinaryOperation {
  cache: [Tensor, Tensor];
  forward(a: Tensor, b: Tensor): Tensor {
    this.cache = [a, b];
    return _pow_tensor(a, b, this);
  }
  backward(dz: Tensor): void {
    const [a, b] = this.cache;
    a.backward(dz.mul(b).mul(a.pow(b.sub(new Tensor(1)))));
    b.backward(dz.mul(a.pow(b)).mul(a.log()));
  }
}

export function _log_tensor(a: Tensor, operation: Operation | null = null): Tensor {
  const kernel = gpu.createKernel(
    function (a: number[]) {
      return Math.log(a[this.thread.x]);
    },
    {
      output: [a.shape.reduce((acc, val) => acc * val, 1)]
    }
  );

  return new Tensor(
    kernel(a.data) as number[],
    { requires_grad: a.requires_grad },
    { operation: operation, shape: a.shape }
  );
}

export class Log extends UnaryOperation {
  cache: [Tensor];
  forward(a: Tensor): Tensor {
    this.cache = [a];
    return _log_tensor(a, this);
  }
  backward(dz: Tensor): void {
    const [a] = this.cache;
    a.backward(new Tensor(1).div(a));
  }
}
