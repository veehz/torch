import torch


def generate_linear_tests():
    tests = []
    # Test standard 2D inputs and batched 3D inputs
    cases = [(10, 5, (3, 10), "2D_input"), (4, 2, (2, 3, 4), "3D_batched_input")]

    for in_features, out_features, input_shape, desc in cases:
        layer = torch.nn.Linear(in_features, out_features)
        x = torch.randn(*input_shape, requires_grad=True)

        out = layer(x)
        out.sum().backward()

        tests.append(
            {
                "test_name": f"linear_{desc}",
                "in_features": in_features,
                "out_features": out_features,
                "input": x.detach().numpy().tolist(),
                "weight": layer.weight.detach().numpy().tolist(),
                "bias": layer.bias.detach().numpy().tolist(),
                "expected_output": out.detach().numpy().tolist(),
                "expected_grad_input": x.grad.numpy().tolist(),
                "expected_grad_weight": layer.weight.grad.numpy().tolist(),
                "expected_grad_bias": layer.bias.grad.numpy().tolist(),
            }
        )
    return tests
