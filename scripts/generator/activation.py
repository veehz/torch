import torch
import torch.nn as nn


def generate_activation_tests():
    tests = []

    cases = [
        # (ActivationClass, test_name, kwargs)
        (nn.ReLU, "relu_1d", (5,), {}),
        (nn.ReLU, "relu_2d", (3, 4), {}),
        (nn.ReLU, "relu_3d", (2, 3, 4), {}),
        (nn.Sigmoid, "sigmoid_1d", (5,), {}),
        (nn.Sigmoid, "sigmoid_2d", (3, 4), {}),
        (nn.Sigmoid, "sigmoid_3d", (2, 3, 4), {}),
        (nn.LeakyReLU, "leaky_relu_default", (5,), {}),
        (nn.LeakyReLU, "leaky_relu_slope_0_2", (3, 4), {"negative_slope": 0.2}),
        (nn.LeakyReLU, "leaky_relu_3d", (2, 3, 4), {"negative_slope": 0.1}),
    ]

    for ActivationClass, desc, input_shape, kwargs in cases:
        input = torch.randn(*input_shape, requires_grad=True)

        activation = ActivationClass(**kwargs)
        output = activation(input)
        output.sum().backward()

        entry = {
            "test_name": desc,
            "activation_type": ActivationClass.__name__,
            "input": input.detach().numpy().tolist(),
            "expected_output": output.detach().numpy().tolist(),
            "expected_grad_input": input.grad.numpy().tolist(),
        }
        if kwargs:
            entry["kwargs"] = kwargs
        tests.append(entry)

    return tests
