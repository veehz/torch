import torch


def generate_binary_tests(op_name, num_tests=3):
    tests = []
    for i in range(num_tests):
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        with torch.no_grad():
            x[0:2, 0] = 0
            x[0:2, 1] = 1
            x[0:2, 2] = -1
            y[0, 0:2] = 0
            y[1, 0:2] = 1
            y[2, 0:2] = -1

        torch_op = getattr(torch, op_name)
        out = torch_op(x, y)
        out.sum().backward()

        tests.append(
            {
                "input_x": x.detach().numpy().tolist(),
                "input_y": y.detach().numpy().tolist(),
                "expected_output": out.detach().numpy().tolist(),
                "expected_grad_x": x.grad.numpy().tolist(),
                "expected_grad_y": y.grad.numpy().tolist(),
            }
        )
    return tests
