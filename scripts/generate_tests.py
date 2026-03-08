import torch
import json

UNARY_OPS = ['log', 'sqrt', 'exp', 'square', 'abs', 'sign', 'neg', 'reciprocal', 'sin', 'cos', 'tan']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow', 'maximum', 'minimum']

torch.manual_seed(42)

def generate_unary_tests(op_name, num_tests=3):
    tests = []
    for i in range(num_tests):
        x = torch.randn(3, 3, requires_grad=True)
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
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

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

if __name__ == "__main__":
    suite = {
        "unary": {op: generate_unary_tests(op) for op in UNARY_OPS},
        "binary": {op: generate_binary_tests(op) for op in BINARY_OPS},
        "broadcasting": generate_broadcasting_tests(),
        "matmul": generate_matmul_tests()
    }

    print("export const testData = ", end="")
    print(json.dumps(suite, indent=2), end=";\n")
