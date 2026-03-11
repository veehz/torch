import torch


def generate_conv_tests():
    tests = []
    cases = [
        (
            "Conv1d",
            torch.nn.Conv1d,
            [
                (1, 1, 3, 1, 0, 1, 1, True, (1, 1, 5), "basic"),
                (2, 3, 2, 2, 1, 1, 1, True, (2, 2, 6), "stride_padding"),
                (2, 2, 2, 1, 0, 2, 1, False, (1, 2, 5), "dilation_no_bias"),
                (4, 4, 3, 1, 1, 1, 2, True, (1, 4, 6), "groups"),
            ],
        ),
        (
            "Conv2d",
            torch.nn.Conv2d,
            [
                (1, 1, 3, 1, 0, 1, 1, True, (1, 1, 5, 5), "basic"),
                (2, 3, 2, 2, 1, 1, 1, True, (2, 2, 6, 6), "stride_padding"),
                (2, 2, 2, 1, 0, 2, 1, False, (1, 2, 5, 5), "dilation_no_bias"),
                (4, 4, 3, 1, 1, 1, 2, True, (1, 4, 4, 4), "groups"),
            ],
        ),
        (
            "Conv3d",
            torch.nn.Conv3d,
            [
                (1, 1, 3, 1, 0, 1, 1, True, (1, 1, 5, 5, 5), "basic"),
                (2, 3, 2, 2, 1, 1, 1, True, (2, 2, 4, 4, 4), "stride_padding"),
                (2, 2, 2, 1, 0, 2, 1, False, (1, 2, 5, 5, 5), "dilation_no_bias"),
                (4, 4, 3, 1, 1, 1, 2, True, (1, 4, 4, 4, 4), "groups"),
            ],
        ),
    ]
    for conv_type, conv_class, conv_cases in cases:
        for (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            input_shape,
            desc,
        ) in conv_cases:
            layer = conv_class(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            x = torch.randn(*input_shape, requires_grad=True)

            out = layer(x)
            out.sum().backward()

            test_data = {
                "test_name": f"{conv_type.lower()}_{desc}",
                "conv_type": conv_type,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
                "has_bias": bias,
                "input": x.detach().numpy().tolist(),
                "weight": layer.weight.detach().numpy().tolist(),
                "expected_output": out.detach().numpy().tolist(),
                "expected_grad_input": x.grad.numpy().tolist(),
                "expected_grad_weight": layer.weight.grad.numpy().tolist(),
            }
            if bias:
                test_data["bias"] = layer.bias.detach().numpy().tolist()
                test_data["expected_grad_bias"] = layer.bias.grad.numpy().tolist()
            else:
                test_data["bias"] = None
                test_data["expected_grad_bias"] = None

            tests.append(test_data)
    return tests
