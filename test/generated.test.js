import { Tensor } from 'torch';
import { assert } from 'chai';
import { testData } from './testcases.gen.js';

function assertDeepCloseTo(actual, expected, delta = 1e-3, first_call = true) {
    if (Array.isArray(expected)) {
        assert.lengthOf(actual, expected.length, 'Array lengths do not match');
        for (let i = 0; i < expected.length; i++) {
            assertDeepCloseTo(actual[i], expected[i], delta, false);
        }
    } else {
        if (Number.isNaN(expected)) {
            assert.isTrue(Number.isNaN(actual), `Expected NaN but got ${actual}`);
        } else {
            assert.closeTo(actual, expected, delta);
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
                        const y = (x)[opName]();
                        assertDeepCloseTo(y.toArray(), test.expected_output);
                        y.sum().backward();
                        assertDeepCloseTo(x.grad.toArray(), test.expected_grad);
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
                        const out = (x)[opName](y);
                        assertDeepCloseTo(out.toArray(), test.expected_output);
                        out.sum().backward();
                        assertDeepCloseTo(x.grad.toArray(), test.expected_grad_x);
                        assertDeepCloseTo(y.grad.toArray(), test.expected_grad_y);
                    });
                });
            });
        }
    });

    describe('Broadcasting Operations', () => {
        testData.broadcasting?.forEach((test) => {
            it(test.test_name, () => {
                const x = new Tensor(test.input_x, { requires_grad: true });
                const y = new Tensor(test.input_y, { requires_grad: true });

                const out = x[test.op_name](y);

                assertDeepCloseTo(out.toArray(), test.expected_output);
                out.sum().backward();
                assertDeepCloseTo(x.grad.toArray(), test.expected_grad_x);
                assertDeepCloseTo(y.grad.toArray(), test.expected_grad_y);
            });
        });
    });

    describe('Matmul Operations', () => {
        testData.matmul?.forEach((test) => {
            it(test.test_name, () => {
                const x = new Tensor(test.input_x, { requires_grad: true });
                const y = new Tensor(test.input_y, { requires_grad: true });

                const out = x.matmul(y);

                assertDeepCloseTo(out.toArray(), test.expected_output);
                out.sum().backward();
                assertDeepCloseTo(x.grad.toArray(), test.expected_grad_x);
                assertDeepCloseTo(y.grad.toArray(), test.expected_grad_y);
            });
        });
    });
});
