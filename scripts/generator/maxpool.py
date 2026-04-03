import torch
import torch.nn as nn


def generate_maxpool_tests():
    tests = []

    cases = [
        ("maxpool_basic_2x2", (1, 1, 4, 4), 2, None, 0),
        ("maxpool_stride_1", (1, 2, 4, 4), 2, 1, 0),
        ("maxpool_padding_1", (1, 1, 4, 4), 3, 1, 1),
        ("maxpool_multichannel", (2, 3, 6, 6), 2, 2, 0),
        ("maxpool_3d_input", (1, 4, 4), 2, None, 0),
    ]

    for desc, shape, kernel_size, stride, padding in cases:
        x = torch.randn(*shape, requires_grad=True)
        if stride is None:
            pool = nn.MaxPool2d(kernel_size, padding=padding)
        else:
            pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        y = pool(x)
        y.sum().backward()

        stride_val = stride if stride is not None else kernel_size

        tests.append(
            {
                "test_name": desc,
                "kernel_size": kernel_size,
                "stride": stride_val,
                "padding": padding,
                "input": x.detach().numpy().tolist(),
                "expected_output": y.detach().numpy().tolist(),
                "expected_grad": x.grad.numpy().tolist(),
            }
        )

    return tests
