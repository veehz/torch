import torch


def generate_broadcasting_tests():
    shape_pairs = [
        ((2, 3), (2, 3)),
        ((), (2, 2)),
        ((2, 3, 4, 1), (3, 1, 1)),
        ((1,), (3, 1, 2)),
        ((5, 1, 4, 1), (3, 1, 1)),
    ]

    ops = ["add", "mul"]
    tests = []

    for shape_x, shape_y in shape_pairs:
        for op_name in ops:
            x = torch.randn(shape_x, requires_grad=True) if shape_x != () else torch.tensor(1.5, requires_grad=True)
            y = torch.randn(shape_y, requires_grad=True) if shape_y != () else torch.tensor(-0.5, requires_grad=True)

            torch_op = getattr(torch, op_name)
            out = torch_op(x, y)

            out.sum().backward()

            sx_str = "scalar" if shape_x == () else str(shape_x)
            sy_str = "scalar" if shape_y == () else str(shape_y)

            tests.append(
                {
                    "test_name": f"broadcast_{op_name}_{sx_str}_and_{sy_str}",
                    "op_name": op_name,
                    "input_x": x.detach().numpy().tolist(),
                    "input_y": y.detach().numpy().tolist(),
                    "expected_output": out.detach().numpy().tolist(),
                    "expected_grad_x": x.grad.numpy().tolist() if x.grad is not None else 0.0,
                    "expected_grad_y": y.grad.numpy().tolist() if y.grad is not None else 0.0,
                }
            )

    return tests
