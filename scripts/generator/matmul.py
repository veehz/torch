import torch


def generate_matmul_tests():
    matmul_cases = [
        ((3,), (3,), "1D_dot_product"),
        ((2, 3), (3, 4), "2D_matrix_multiply"),
        ((3,), (3, 4), "1D_2D_prepend_remove"),
        ((2, 3), (3,), "2D_1D_matrix_vector"),
        ((2, 1, 2, 3), (3, 3, 2), "ND_batched_with_broadcast"),
    ]

    tests = []

    for shape_x, shape_y, desc in matmul_cases:
        x = torch.randn(shape_x, requires_grad=True)
        y = torch.randn(shape_y, requires_grad=True)

        out = torch.matmul(x, y)
        out.sum().backward()

        tests.append(
            {
                "test_name": f"matmul_{desc}",
                "input_x": x.detach().numpy().tolist(),
                "input_y": y.detach().numpy().tolist(),
                "expected_output": out.detach().numpy().tolist(),
                "expected_grad_x": x.grad.numpy().tolist(),
                "expected_grad_y": y.grad.numpy().tolist(),
            }
        )

    return tests
