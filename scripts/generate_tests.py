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

if __name__ == "__main__":
    suite = {
        "unary": {op: generate_unary_tests(op) for op in UNARY_OPS},
        "binary": {op: generate_binary_tests(op) for op in BINARY_OPS}
    }

    print("export const testData = ", end="")
    print(json.dumps(suite, indent=2), end=";\n")