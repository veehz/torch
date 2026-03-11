import torch

from generator.encoder import CompactJSONEncoder
from generator.unary import generate_unary_tests
from generator.binary import generate_binary_tests
from generator.broadcasting import generate_broadcasting_tests
from generator.matmul import generate_matmul_tests
from generator.reduction import generate_reduction_tests
from generator.linear import generate_linear_tests
from generator.optimizer import generate_optimizer_tests
from generator.expand import generate_expand_tests
from generator.conv import generate_conv_tests
from generator.export import generate_export_tests
from generator.loss import generate_loss_tests
from generator.activation import generate_activation_tests

UNARY_OPS = ["log", "sqrt", "exp", "square", "abs", "sign", "neg", "reciprocal", "sin", "cos", "tan"]
BINARY_OPS = ["add", "sub", "mul", "div", "pow", "maximum", "minimum"]

torch.manual_seed(42)

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
        "export": generate_export_tests(),
        "loss": generate_loss_tests(),
        "activations": generate_activation_tests(),
    }

    print("export const testData = ", end="")
    print(CompactJSONEncoder(indent=2).encode(suite), end=";\n")
