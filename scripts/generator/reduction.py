import torch


def generate_reduction_tests():
    tests = []
    ops = ["sum", "mean", "max", "min"]
    dims = [None, 0, 1, -1]
    keepdims = [False, True]

    for op in ops:
        for dim in dims:
            for keepdim in keepdims:
                if dim is None and keepdim:
                    continue

                x = torch.randn(3, 4, 5, requires_grad=True)
                torch_op = getattr(torch, op)

                try:
                    if dim is None:
                        y = torch_op(x)
                    else:
                        out = torch_op(x, dim=dim, keepdim=keepdim)
                        y = out.values if op in ["max", "min"] else out

                    y.sum().backward()

                    tests.append(
                        {
                            "test_name": f"{op}_dim_{dim}_keepdim_{keepdim}",
                            "op_name": op,
                            "dim": dim,
                            "keepdim": keepdim,
                            "input": x.detach().numpy().tolist(),
                            "expected_output": y.detach().numpy().tolist(),
                            "expected_grad": x.grad.numpy().tolist(),
                        }
                    )
                except Exception as e:
                    pass
    return tests
