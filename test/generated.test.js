import * as torch from 'torch';
import { Tensor } from 'torch';
import { assert } from 'chai';
import { testData } from './testcases.gen.js';

function assertDeepCloseTo(actual, expected, name = null, delta = 1e-3, first_call = true) {
  if (Array.isArray(expected)) {
    assert.lengthOf(actual, expected.length, 'Array lengths do not match');
    for (let i = 0; i < expected.length; i++) {
      assertDeepCloseTo(actual[i], expected[i], name, delta, false);
    }
  } else {
    if (Number.isNaN(expected)) {
      assert.isTrue(Number.isNaN(actual), `${name}: Expected NaN but got ${actual}`);
    } else if(!Number.isFinite(expected)) {
      assert.equal(actual, expected, `${name}: Expected ${expected} but got ${actual}`);
    } else {
      assert.closeTo(actual, expected, delta, `${name}: Expected ${expected} but got ${actual}`);
    }
  }
}

describe('Automated Tests', () => {
  describe('Unary Operations', () => {
    for (const [opName, tests] of Object.entries(testData.unary)) {
      describe(`.${opName}()`, () => {
        tests.forEach((test, idx) => {
          it(`case ${idx + 1}`, () => {
            const x = new Tensor(test.input, { requires_grad: true });
            const y = x[opName]();
            assertDeepCloseTo(y.toArray(), test.expected_output, `${opName} output`);
            y.sum().backward();
            assertDeepCloseTo(x.grad.toArray(), test.expected_grad, `${opName} grad`);
          });
        });
      });
    }
  });

  describe('Binary Operations', () => {
    for (const [opName, tests] of Object.entries(testData.binary)) {
      describe(`.${opName}()`, () => {
        tests.forEach((test, idx) => {
          it(`case ${idx + 1}`, () => {
            const x = new Tensor(test.input_x, { requires_grad: true });
            const y = new Tensor(test.input_y, { requires_grad: true });
            const out = x[opName](y);
            assertDeepCloseTo(out.toArray(), test.expected_output, `${opName} output`);
            out.sum().backward();
            assertDeepCloseTo(x.grad.toArray(), test.expected_grad_x, `${opName} grad x`);
            assertDeepCloseTo(y.grad.toArray(), test.expected_grad_y, `${opName} grad y`);
          });
        });
      });
    }
  });

  describe('Broadcasting Operations', () => {
    testData.broadcasting?.forEach(test => {
      it(test.test_name, () => {
        const x = new Tensor(test.input_x, { requires_grad: true });
        const y = new Tensor(test.input_y, { requires_grad: true });

        const out = x[test.op_name](y);

        assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);
        out.sum().backward();
        assertDeepCloseTo(x.grad.toArray(), test.expected_grad_x, `${test.test_name} grad x`);
        assertDeepCloseTo(y.grad.toArray(), test.expected_grad_y, `${test.test_name} grad y`);
      });
    });
  });

  describe('Matmul Operations', () => {
    testData.matmul?.forEach(test => {
      it(test.test_name, () => {
        const x = new Tensor(test.input_x, { requires_grad: true });
        const y = new Tensor(test.input_y, { requires_grad: true });

        const out = x.matmul(y);

        assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);
        out.sum().backward();
        assertDeepCloseTo(x.grad.toArray(), test.expected_grad_x, `${test.test_name} grad x`);
        assertDeepCloseTo(y.grad.toArray(), test.expected_grad_y, `${test.test_name} grad y`);
      });
    });
  });

  describe('Reduction Operations', () => {
    testData.reductions?.forEach(test => {
      it(test.test_name, () => {
        const x = new Tensor(test.input, { requires_grad: true });
        let out;
        if (test.dim === null || test.dim === undefined) {
          out = x[test.op_name]();
        } else {
          out = x[test.op_name](test.dim, test.keepdim);
        }
        assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);
        out.sum().backward();
        assertDeepCloseTo(x.grad.toArray(), test.expected_grad, `${test.test_name} grad`);
      });
    });
  });

  describe('Neural Network Modules', () => {
    describe('nn.Linear', () => {
      testData.linear?.forEach(test => {
        it(test.test_name, () => {
          const layer = new torch.nn.Linear(test.in_features, test.out_features);

          // Overwrite the layer's internal parameters with Python's starting state
          layer.weight = new Tensor(test.weight, { requires_grad: true });
          layer.bias = new Tensor(test.bias, { requires_grad: true });

          const x = new Tensor(test.input, { requires_grad: true });

          const out = layer.forward(x);
          assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);

          out.sum().backward();

          assertDeepCloseTo(x.grad.toArray(), test.expected_grad_input, `${test.test_name} grad input`);
          assertDeepCloseTo(layer.weight.grad.toArray(), test.expected_grad_weight, `${test.test_name} grad weight`);
          assertDeepCloseTo(layer.bias.grad.toArray(), test.expected_grad_bias, `${test.test_name} grad bias`);
        });
      });
    });

    describe('Convolutions', () => {
      testData.conv?.forEach(test => {
        it(test.test_name, () => {
          const ConvClass = torch.nn[test.conv_type];
          const layer = new ConvClass(
            test.in_channels,
            test.out_channels,
            test.kernel_size,
            test.stride,
            test.padding,
            test.dilation,
            test.groups,
            test.has_bias
          );

          layer.weight = new Tensor(test.weight, { requires_grad: true });
          if (test.has_bias) {
            layer.bias = new Tensor(test.bias, { requires_grad: true });
          }

          const x = new Tensor(test.input, { requires_grad: true });

          const out = layer.forward(x);
          assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);

          out.sum().backward();

          assertDeepCloseTo(x.grad.toArray(), test.expected_grad_input, `${test.test_name} grad input`);
          assertDeepCloseTo(layer.weight.grad.toArray(), test.expected_grad_weight, `${test.test_name} grad weight`);
          if (test.has_bias) {
            assertDeepCloseTo(layer.bias.grad.toArray(), test.expected_grad_bias, `${test.test_name} grad bias`);
          }
        });
      });
    });
  });

  describe('Optimizers', () => {
    testData.optimizers?.forEach(test => {
      it(test.test_name, () => {
        const w = new torch.nn.Parameter(test.initial_weight, { requires_grad: true });
        const x = new Tensor(test.input_x);

        const OptimizerClass = torch.optim[test.optimizer];
        let optimizer;

        if (test.optimizer === 'SGD') {
          optimizer = new OptimizerClass(
            [w],
            test.kwargs.lr,
            test.kwargs.momentum,
            test.kwargs.dampening,
            test.kwargs.weight_decay,
            test.kwargs.nesterov,
            test.kwargs.maximize
          );
        } else if (test.optimizer === 'Adam') {
          optimizer = new OptimizerClass(
            [w],
            test.kwargs.lr,
            test.kwargs.betas,
            test.kwargs.eps,
            test.kwargs.weight_decay,
            test.kwargs.amsgrad,
            test.kwargs.maximize
          );
        }

        optimizer.zero_grad();
        const loss = w.mul(x).sum();
        loss.backward();

        assertDeepCloseTo(w.grad.toArray(), test.expected_grad, `${test.test_name} grad`);

        optimizer.step();

        assertDeepCloseTo(w.toArray(), test.expected_updated_weight, `${test.test_name} updated weight`);
      });
    });
  });

  describe('Expand Operations', () => {
    testData.expand?.forEach(test => {
      it(test.test_name, () => {
        const x = new Tensor(test.input, { requires_grad: true });
        const out = x.expand(test.expand_shape);
        assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);
        out.sum().backward();
        assertDeepCloseTo(x.grad.toArray(), test.expected_grad, `${test.test_name} grad`);
      });
    });
  });
});
