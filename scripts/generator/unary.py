import torch


def generate_unary_tests(op_name, num_tests=3):
    tests = []
    for i in range(num_tests):
        x = torch.randn(3, 3, requires_grad=True)
        with torch.no_grad():
            x[0, 0] = 0
            x[0, 1] = 1
            x[0, 2] = -1

        torch_op = getattr(torch, op_name)
        y = torch_op(x)
        y.sum().backward()

        tests.append(
            {
                "input": x.detach().numpy().tolist(),
                "expected_output": y.detach().numpy().tolist(),
                "expected_grad": x.grad.numpy().tolist(),
            }
        )
    return tests
