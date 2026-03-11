import torch
import torch.nn as nn


def generate_loss_tests():
    tests = []

    cases = [
        # (LossClass, input_shape, target_shape, test_name, needs_sigmoid)
        (nn.MSELoss, (5,), (5,), "mse_1d", False),
        (nn.MSELoss, (3, 4), (3, 4), "mse_2d", False),
        (nn.MSELoss, (2, 3, 4), (2, 3, 4), "mse_3d", False),
        (nn.L1Loss, (5,), (5,), "l1_1d", False),
        (nn.L1Loss, (3, 4), (3, 4), "l1_2d", False),
        (nn.L1Loss, (2, 3, 4), (2, 3, 4), "l1_3d", False),
        # BCE requires inputs in (0, 1)
        (nn.BCELoss, (5,), (5,), "bce_1d", True),
        (nn.BCELoss, (3, 4), (3, 4), "bce_2d", True),
    ]

    for LossClass, input_shape, target_shape, desc, needs_sigmoid in cases:
        if needs_sigmoid:
            input = torch.sigmoid(torch.randn(*input_shape)).requires_grad_(True)
            target = torch.rand(*target_shape)
        else:
            input = torch.randn(*input_shape, requires_grad=True)
            target = torch.randn(*target_shape)

        loss_fn = LossClass()
        output = loss_fn(input, target)
        output.backward()

        tests.append(
            {
                "test_name": desc,
                "loss_type": LossClass.__name__,
                "input": input.detach().numpy().tolist(),
                "target": target.detach().numpy().tolist(),
                "expected_output": output.detach().numpy().tolist(),
                "expected_grad_input": input.grad.numpy().tolist(),
            }
        )

    return tests
