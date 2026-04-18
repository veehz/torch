import * as torch from '@sourceacademy/torch';
import { Tensor } from '@sourceacademy/torch';
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

    describe('Loss Functions', () => {
      testData.loss?.forEach(test => {
        it(test.test_name, () => {
          const LossClass = torch.nn[test.loss_type];
          const loss_fn = new LossClass();

          const input = new Tensor(test.input, { requires_grad: true });
          const target = new Tensor(test.target);

          const output = loss_fn.forward(input, target);
          assertDeepCloseTo(output.toArray(), test.expected_output, `${test.test_name} output`);

          output.backward();
          assertDeepCloseTo(input.grad.toArray(), test.expected_grad_input, `${test.test_name} grad input`);
        });
      });
    });

    describe('Activation Functions', () => {
      testData.activations?.forEach(test => {
        it(test.test_name, () => {
          const ActivationClass = torch.nn[test.activation_type];
          const kwargs = test.kwargs || {};
          const activation = test.activation_type === 'LeakyReLU'
            ? new ActivationClass(kwargs.negative_slope)
            : new ActivationClass();

          const input = new Tensor(test.input, { requires_grad: true });

          const output = activation.forward(input);
          assertDeepCloseTo(output.toArray(), test.expected_output, `${test.test_name} output`);

          output.sum().backward();
          assertDeepCloseTo(input.grad.toArray(), test.expected_grad_input, `${test.test_name} grad input`);
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
        } else if (test.optimizer === 'Adagrad') {
          optimizer = new OptimizerClass(
            [w],
            test.kwargs.lr,
            test.kwargs.lr_decay,
            test.kwargs.weight_decay,
            test.kwargs.eps
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

  describe('Cat Operations', () => {
    testData.cat?.forEach(test => {
      it(test.test_name, () => {
        const tensors = test.inputs.map(inp => new Tensor(inp, { requires_grad: true }));
        const out = torch.cat(tensors, test.dim);
        assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);
        out.sum().backward();
        tensors.forEach((t, i) => {
          assertDeepCloseTo(t.grad.toArray(), test.expected_grads[i], `${test.test_name} grad[${i}]`);
        });
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
  describe('Softmax', () => {
    testData.softmax?.forEach(test => {
      it(test.test_name, () => {
        const x = new Tensor(test.input, { requires_grad: true });
        const out = torch.softmax(x, test.dim);
        assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);
        out.sum().backward();
        assertDeepCloseTo(x.grad.toArray(), test.expected_grad, `${test.test_name} grad`);
      });
    });
  });

  describe('Clamp', () => {
    testData.clamp?.forEach(test => {
      it(test.test_name, () => {
        const x = new Tensor(test.input, { requires_grad: true });
        const out = torch.clamp(x, test.min, test.max);
        assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);
        out.sum().backward();
        assertDeepCloseTo(x.grad.toArray(), test.expected_grad, `${test.test_name} grad`);
      });
    });
  });

  describe('MaxPool2d', () => {
    testData.maxpool?.forEach(test => {
      it(test.test_name, () => {
        const x = new Tensor(test.input, { requires_grad: true });
        const pool = new torch.nn.MaxPool2d(test.kernel_size, test.stride, test.padding);
        const out = pool.forward(x);
        assertDeepCloseTo(out.toArray(), test.expected_output, `${test.test_name} output`);
        out.sum().backward();
        assertDeepCloseTo(x.grad.toArray(), test.expected_grad, `${test.test_name} grad`);
      });
    });
  });

  describe('Export', () => {
    testData.export?.forEach(test => {
      it(`export_${test.test_name}`, () => {
        let model;

        if (test.model_type === 'LinearReLU') {
          const linear = new torch.nn.Linear(test.in_features, test.out_features);
          linear.weight = new Tensor(test.weight, { requires_grad: true });
          linear.bias = new Tensor(test.bias, { requires_grad: true });
          const relu = new torch.nn.ReLU();
          model = new torch.nn.Sequential(linear, relu);
        } else if (test.model_type === 'TwoLayer') {
          const linear1 = new torch.nn.Linear(test.linear1_in, test.linear1_out);
          linear1.weight = new Tensor(test.linear1_weight, { requires_grad: true });
          linear1.bias = new Tensor(test.linear1_bias, { requires_grad: true });
          const relu = new torch.nn.ReLU();
          const linear2 = new torch.nn.Linear(test.linear2_in, test.linear2_out);
          linear2.weight = new Tensor(test.linear2_weight, { requires_grad: true });
          linear2.bias = new Tensor(test.linear2_bias, { requires_grad: true });
          const sigmoid = new torch.nn.Sigmoid();
          model = new torch.nn.Sequential(linear1, relu, linear2, sigmoid);
        }

        const x = new Tensor(test.input);
        const ep = torch.export_(model, [x]);

        // Verify graph structure: must have placeholder, call_function, and output nodes
        const placeholders = ep.graph.filter(n => n.op === 'placeholder');
        const callFunctions = ep.graph.filter(n => n.op === 'call_function');
        const outputs = ep.graph.filter(n => n.op === 'output');

        assert.isAbove(placeholders.length, 0, 'Should have placeholder nodes');
        assert.isAbove(callFunctions.length, 0, 'Should have call_function nodes');
        assert.equal(outputs.length, 1, 'Should have exactly one output node');

        // Verify all call_function nodes have aten.* targets
        for (const node of callFunctions) {
          assert.match(node.target, /^aten\./, `Target should start with aten.: ${node.target}`);
        }

        // Verify graph signature matches PyTorch's
        // Input specs: parameters should be PARAMETER, user inputs should be USER_INPUT
        const expectedParamSpecs = test.expected_input_specs.filter(s => s.kind === 'PARAMETER');
        const expectedUserSpecs = test.expected_input_specs.filter(s => s.kind === 'USER_INPUT');

        const actualParamSpecs = ep.graph_signature.input_specs.filter(s => s.kind === 'PARAMETER');
        const actualUserSpecs = ep.graph_signature.input_specs.filter(s => s.kind === 'USER_INPUT');

        assert.equal(actualParamSpecs.length, expectedParamSpecs.length, 'Number of parameter specs should match');
        assert.equal(actualUserSpecs.length, expectedUserSpecs.length, 'Number of user input specs should match');

        // Verify parameter placeholder naming matches PyTorch convention
        for (let i = 0; i < expectedParamSpecs.length; i++) {
          assert.equal(actualParamSpecs[i].name, expectedParamSpecs[i].name,
            `Parameter spec name should match: expected ${expectedParamSpecs[i].name}`);
          assert.equal(actualParamSpecs[i].kind, 'PARAMETER');
        }

        // Verify user input naming
        for (let i = 0; i < expectedUserSpecs.length; i++) {
          assert.equal(actualUserSpecs[i].name, expectedUserSpecs[i].name,
            `User input spec name should match: expected ${expectedUserSpecs[i].name}`);
        }

        // Verify output spec
        assert.equal(ep.graph_signature.output_specs.length, test.expected_output_specs.length);
        assert.equal(ep.graph_signature.output_specs[0].kind, 'USER_OUTPUT');

        // Verify placeholder node shapes match
        const expectedPlaceholders = test.expected_nodes.filter(n => n.op === 'placeholder');
        for (let i = 0; i < expectedPlaceholders.length; i++) {
          assert.equal(placeholders[i].name, expectedPlaceholders[i].name,
            `Placeholder name should match: expected ${expectedPlaceholders[i].name}`);
          if (expectedPlaceholders[i].val_shape) {
            assert.deepEqual(placeholders[i].val_shape, expectedPlaceholders[i].val_shape,
              `Placeholder shape should match for ${expectedPlaceholders[i].name}`);
          }
        }

        // Verify output node references a valid graph node
        const outputArgs = outputs[0].args;
        assert.isAbove(outputArgs.length, 0, 'Output should reference at least one node');

        // All node names should be unique
        const allNames = ep.graph.map(n => n.name);
        const uniqueNames = new Set(allNames);
        assert.equal(allNames.length, uniqueNames.size, 'All node names should be unique');
      });
    });
  });
});
