import torch
import torch.nn.functional as F


def generate_softmax_tests():
    tests = []

    cases = [
        ("softmax_1d_dim0", (5,), 0),
        ("softmax_2d_dim0", (3, 4), 0),
        ("softmax_2d_dim1", (3, 4), 1),
        ("softmax_3d_dim1", (2, 3, 4), 1),
        ("softmax_3d_dim2", (2, 3, 4), 2),
    ]

    for desc, shape, dim in cases:
        x = torch.randn(*shape, requires_grad=True)
        y = F.softmax(x, dim=dim)
        y.sum().backward()

        tests.append(
            {
                "test_name": desc,
                "dim": dim,
                "input": x.detach().numpy().tolist(),
                "expected_output": y.detach().numpy().tolist(),
                "expected_grad": x.grad.numpy().tolist(),
            }
        )

    return tests
