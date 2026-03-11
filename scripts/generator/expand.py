import torch


def generate_expand_tests():
    tests = []

    # Tuples of (initial_shape, expand_shape, description)
    cases = [
        ((1,), (3,), "1D_expand"),
        ((3,), (2, 3), "prepend_2D"),
        ((1, 3), (4, 3), "expand_dim_0"),
        ((2, 1, 4), (2, 5, 4), "expand_middle_dim"),
        ((1, 3, 1), (2, -1, 4), "preserve_with_negative_one"),
    ]

    for initial_shape, expand_shape, desc in cases:
        x = torch.randn(initial_shape, requires_grad=True)

        out = x.expand(*expand_shape)
        out.sum().backward()

        tests.append(
            {
                "test_name": f"expand_{desc}",
                "input": x.detach().numpy().tolist(),
                "expand_shape": list(expand_shape),
                "expected_output": out.detach().numpy().tolist(),
                "expected_grad": x.grad.numpy().tolist(),
            }
        )

    return tests
