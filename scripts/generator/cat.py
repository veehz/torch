import torch


def generate_cat_tests():
    tests = []

    # (list_of_shapes, dim)
    configs = [
        ([(3,), (4,)], 0),                    # 1D, two tensors
        ([(3,), (4,), (2,)], 0),               # 1D, three tensors
        ([(2, 3), (4, 3)], 0),                 # 2D, along dim 0
        ([(2, 3), (2, 4)], 1),                 # 2D, along dim 1
        ([(2, 3, 4), (5, 3, 4)], 0),           # 3D, along dim 0
        ([(2, 3, 4), (2, 1, 4)], 1),           # 3D, along dim 1
        ([(2, 3, 4), (2, 3, 2)], 2),           # 3D, along dim 2
        ([(2, 3, 4), (2, 3, 4)], -1),          # 3D, negative dim
        ([(2, 3), (2, 3), (2, 3)], 0),         # three equal-shape tensors
    ]

    for shapes, dim in configs:
        tensors = [torch.randn(*s, requires_grad=True) for s in shapes]
        out = torch.cat(tensors, dim=dim)
        out.sum().backward()

        name_parts = "_".join("x".join(str(d) for d in s) for s in shapes)
        tests.append({
            "test_name": f"cat_dim{dim}_{name_parts}",
            "inputs": [t.detach().numpy().tolist() for t in tensors],
            "dim": dim,
            "expected_output": out.detach().numpy().tolist(),
            "expected_grads": [t.grad.numpy().tolist() for t in tensors],
        })

    return tests
