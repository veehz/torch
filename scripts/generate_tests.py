import torch
import json

UNARY_OPS = ['log', 'sqrt', 'exp', 'square', 'abs', 'sign', 'neg', 'reciprocal', 'sin', 'cos', 'tan']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow', 'maximum', 'minimum']

torch.manual_seed(42)

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

        tests.append({
            "input": x.detach().numpy().tolist(),
            "expected_output": y.detach().numpy().tolist(),
            "expected_grad": x.grad.numpy().tolist()
        })
    return tests

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

        tests.append({
            "input_x": x.detach().numpy().tolist(),
            "input_y": y.detach().numpy().tolist(),
            "expected_output": out.detach().numpy().tolist(),
            "expected_grad_x": x.grad.numpy().tolist(),
            "expected_grad_y": y.grad.numpy().tolist()
        })
    return tests

def generate_broadcasting_tests():
    shape_pairs = [
        ((2, 3), (2, 3)),
        ((), (2, 2)),
        ((2, 3, 4, 1), (3, 1, 1)),
        ((1,), (3, 1, 2)),
        ((5, 1, 4, 1), (3, 1, 1))
    ]

    ops = ['add', 'mul']
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

            tests.append({
                "test_name": f"broadcast_{op_name}_{sx_str}_and_{sy_str}",
                "op_name": op_name,
                "input_x": x.detach().numpy().tolist(),
                "input_y": y.detach().numpy().tolist(),
                "expected_output": out.detach().numpy().tolist(),
                "expected_grad_x": x.grad.numpy().tolist() if x.grad is not None else 0.0,
                "expected_grad_y": y.grad.numpy().tolist() if y.grad is not None else 0.0
            })

    return tests

def generate_matmul_tests():
    matmul_cases = [
        ((3,), (3,), "1D_dot_product"),
        ((2, 3), (3, 4), "2D_matrix_multiply"),
        ((3,), (3, 4), "1D_2D_prepend_remove"),
        ((2, 3), (3,), "2D_1D_matrix_vector"),
        ((2, 1, 2, 3), (3, 3, 2), "ND_batched_with_broadcast")
    ]

    tests = []

    for shape_x, shape_y, desc in matmul_cases:
        x = torch.randn(shape_x, requires_grad=True)
        y = torch.randn(shape_y, requires_grad=True)

        out = torch.matmul(x, y)
        out.sum().backward()

        tests.append({
            "test_name": f"matmul_{desc}",
            "input_x": x.detach().numpy().tolist(),
            "input_y": y.detach().numpy().tolist(),
            "expected_output": out.detach().numpy().tolist(),
            "expected_grad_x": x.grad.numpy().tolist(),
            "expected_grad_y": y.grad.numpy().tolist()
        })

    return tests

def generate_reduction_tests():
    tests = []
    ops = ['sum', 'mean', 'max', 'min']
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
                        y = out.values if op in ['max', 'min'] else out

                    y.sum().backward()

                    tests.append({
                        "test_name": f"{op}_dim_{dim}_keepdim_{keepdim}",
                        "op_name": op,
                        "dim": dim,
                        "keepdim": keepdim,
                        "input": x.detach().numpy().tolist(),
                        "expected_output": y.detach().numpy().tolist(),
                        "expected_grad": x.grad.numpy().tolist()
                    })
                except Exception as e:
                    pass
    return tests

def generate_linear_tests():
    tests = []
    # Test standard 2D inputs and batched 3D inputs
    cases = [
        (10, 5, (3, 10), "2D_input"), 
        (4, 2, (2, 3, 4), "3D_batched_input")
    ]
    
    for in_features, out_features, input_shape, desc in cases:
        layer = torch.nn.Linear(in_features, out_features)
        x = torch.randn(*input_shape, requires_grad=True)

        out = layer(x)
        out.sum().backward()

        tests.append({
            "test_name": f"linear_{desc}",
            "in_features": in_features,
            "out_features": out_features,
            "input": x.detach().numpy().tolist(),
            "weight": layer.weight.detach().numpy().tolist(),
            "bias": layer.bias.detach().numpy().tolist(),
            "expected_output": out.detach().numpy().tolist(),
            "expected_grad_input": x.grad.numpy().tolist(),
            "expected_grad_weight": layer.weight.grad.numpy().tolist(),
            "expected_grad_bias": layer.bias.grad.numpy().tolist()
        })
    return tests

def generate_optimizer_tests():
    tests = []
    configs = [
        ("SGD_basic", torch.optim.SGD, {"lr": 0.1}),
        ("SGD_momentum", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}),
        ("Adam_basic", torch.optim.Adam, {"lr": 0.1}),
        ("Adam_custom_betas", torch.optim.Adam, {"lr": 0.1, "betas": (0.95, 0.999)})
    ]

    for test_name, optim_class, kwargs in configs:
        # 1. Initialize a generic parameter tensor
        w = torch.randn(3, 3, requires_grad=True)
        x = torch.randn(3, 3) # Dummy input to calculate a loss
        
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

        tests.append({
            "test_name": test_name,
            "optimizer": test_name.split("_")[0], # "SGD" or "Adam"
            "kwargs": kwargs,
            "initial_weight": initial_w.numpy().tolist(),
            "input_x": x.numpy().tolist(),
            "expected_grad": expected_grad.numpy().tolist(),
            "expected_updated_weight": w.detach().numpy().tolist()
        })
        
    return tests

def generate_expand_tests():
    tests = []
    
    # Tuples of (initial_shape, expand_shape, description)
    cases = [
        ((1,), (3,), "1D_expand"),
        ((3,), (2, 3), "prepend_2D"),
        ((1, 3), (4, 3), "expand_dim_0"),
        ((2, 1, 4), (2, 5, 4), "expand_middle_dim"),
        ((1, 3, 1), (2, -1, 4), "preserve_with_negative_one")
    ]
    
    for initial_shape, expand_shape, desc in cases:
        x = torch.randn(initial_shape, requires_grad=True)
        
        out = x.expand(*expand_shape)
        out.sum().backward()
        
        tests.append({
            "test_name": f"expand_{desc}",
            "input": x.detach().numpy().tolist(),
            "expand_shape": list(expand_shape),
            "expected_output": out.detach().numpy().tolist(),
            "expected_grad": x.grad.numpy().tolist()
        })
        
    return tests

def generate_conv_tests():
    tests = []
    cases = [
        ("Conv1d", torch.nn.Conv1d, [
            (1, 1, 3, 1, 0, 1, 1, True, (1, 1, 5), "basic"),
            (2, 3, 2, 2, 1, 1, 1, True, (2, 2, 6), "stride_padding"),
            (2, 2, 2, 1, 0, 2, 1, False, (1, 2, 5), "dilation_no_bias"),
            (4, 4, 3, 1, 1, 1, 2, True, (1, 4, 6), "groups"),
        ]),
        ("Conv2d", torch.nn.Conv2d, [
            (1, 1, 3, 1, 0, 1, 1, True, (1, 1, 5, 5), "basic"),
            (2, 3, 2, 2, 1, 1, 1, True, (2, 2, 6, 6), "stride_padding"),
            (2, 2, 2, 1, 0, 2, 1, False, (1, 2, 5, 5), "dilation_no_bias"),
            (4, 4, 3, 1, 1, 1, 2, True, (1, 4, 4, 4), "groups"),
        ]),
        ("Conv3d", torch.nn.Conv3d, [
            (1, 1, 3, 1, 0, 1, 1, True, (1, 1, 5, 5, 5), "basic"),
            (2, 3, 2, 2, 1, 1, 1, True, (2, 2, 4, 4, 4), "stride_padding"),
            (2, 2, 2, 1, 0, 2, 1, False, (1, 2, 5, 5, 5), "dilation_no_bias"),
            (4, 4, 3, 1, 1, 1, 2, True, (1, 4, 4, 4, 4), "groups"),
        ])
    ]
    for conv_type, conv_class, conv_cases in cases:
        for in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, input_shape, desc in conv_cases:
            layer = conv_class(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
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
                "expected_grad_weight": layer.weight.grad.numpy().tolist()
            }
            if bias:
                test_data["bias"] = layer.bias.detach().numpy().tolist()
                test_data["expected_grad_bias"] = layer.bias.grad.numpy().tolist()
            else:
                test_data["bias"] = None
                test_data["expected_grad_bias"] = None

            tests.append(test_data)
    return tests

if __name__ == "__main__":
    suite = {
        "unary": {op: generate_unary_tests(op) for op in UNARY_OPS},
        "binary": {op: generate_binary_tests(op) for op in BINARY_OPS},
        "broadcasting": generate_broadcasting_tests(),
        "matmul": generate_matmul_tests(),
        "reductions": generate_reduction_tests(),
        "linear": generate_linear_tests(),
        "optimizers": generate_optimizer_tests(),
        "expand": generate_expand_tests(),
        "conv": generate_conv_tests(),
    }

    print("export const testData = ", end="")
    print(json.dumps(suite, indent=2), end=";\n")
