import torch
import torch.nn as nn


def generate_activation_tests():
    tests = []

    cases = [
        # (ActivationClass, test_name)
        (nn.ReLU, "relu_1d", (5,)),
        (nn.ReLU, "relu_2d", (3, 4)),
        (nn.ReLU, "relu_3d", (2, 3, 4)),
        (nn.Sigmoid, "sigmoid_1d", (5,)),
        (nn.Sigmoid, "sigmoid_2d", (3, 4)),
        (nn.Sigmoid, "sigmoid_3d", (2, 3, 4)),
    ]

    for ActivationClass, desc, input_shape in cases:
        input = torch.randn(*input_shape, requires_grad=True)

        activation = ActivationClass()
        output = activation(input)
        output.sum().backward()

        tests.append(
            {
                "test_name": desc,
                "activation_type": ActivationClass.__name__,
                "input": input.detach().numpy().tolist(),
                "expected_output": output.detach().numpy().tolist(),
                "expected_grad_input": input.grad.numpy().tolist(),
            }
        )

    return tests
