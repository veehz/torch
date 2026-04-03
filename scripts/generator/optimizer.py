import torch


def generate_optimizer_tests():
    tests = []
    configs = [
        ("SGD_basic", torch.optim.SGD, {"lr": 0.1}),
        ("SGD_momentum", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}),
        ("SGD_weight_decay", torch.optim.SGD, {"lr": 0.1, "weight_decay": 0.01}),
        ("SGD_nesterov", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "nesterov": True}),
        ("Adam_basic", torch.optim.Adam, {"lr": 0.1}),
        ("Adam_custom_betas", torch.optim.Adam, {"lr": 0.1, "betas": (0.95, 0.999)}),
        ("Adam_weight_decay", torch.optim.Adam, {"lr": 0.1, "weight_decay": 0.01}),
        ("Adam_amsgrad", torch.optim.Adam, {"lr": 0.1, "amsgrad": True}),
        ("Adagrad_basic", torch.optim.Adagrad, {"lr": 0.1}),
        ("Adagrad_lr_decay", torch.optim.Adagrad, {"lr": 0.1, "lr_decay": 0.01}),
        ("Adagrad_weight_decay", torch.optim.Adagrad, {"lr": 0.1, "weight_decay": 0.01}),
    ]

    for test_name, optim_class, kwargs in configs:
        # 1. Initialize a generic parameter tensor
        w = torch.randn(3, 3, requires_grad=True)
        x = torch.randn(3, 3)  # Dummy input to calculate a loss

        # Save exact starting state
        initial_w = w.detach().clone()

        # 2. Setup Optimizer
        optimizer = optim_class([w], **kwargs)
        optimizer.zero_grad()

        # 3. Compute loss and gradients
        loss = (w * x).sum()
        loss.backward()

        # Save exact gradient computed by PyTorch
        expected_grad = w.grad.detach().clone()

        # 4. Step the optimizer
        optimizer.step()

        tests.append(
            {
                "test_name": test_name,
                "optimizer": test_name.split("_")[0],  # "SGD" or "Adam"
                "kwargs": kwargs,
                "initial_weight": initial_w.numpy().tolist(),
                "input_x": x.numpy().tolist(),
                "expected_grad": expected_grad.numpy().tolist(),
                "expected_updated_weight": w.detach().numpy().tolist(),
            }
        )

    return tests
