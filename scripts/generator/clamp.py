import torch


def generate_clamp_tests():
    tests = []

    cases = [
        ("clamp_1d", (5,), -0.5, 0.5),
        ("clamp_2d", (3, 4), 0.0, 1.0),
        ("clamp_3d", (2, 3, 4), -1.0, 1.0),
        ("clamp_negative_range", (4,), -2.0, -0.5),
    ]

    for desc, shape, min_val, max_val in cases:
        x = torch.randn(*shape, requires_grad=True)
        y = torch.clamp(x, min=min_val, max=max_val)
        y.sum().backward()

        tests.append(
            {
                "test_name": desc,
                "min": min_val,
                "max": max_val,
                "input": x.detach().numpy().tolist(),
                "expected_output": y.detach().numpy().tolist(),
                "expected_grad": x.grad.numpy().tolist(),
            }
        )

    return tests
