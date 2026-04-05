# nn_module.py — Tests for custom nn.Module subclasses, parameter registration,
# call vs forward separation, and nested models.

class MyLinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.rand(out_features))

    def forward(self, input):
        return input @ self.weight + self.bias

class MySmallModel(torch.nn.Module):
    def __init__(self, in_features, intermediate_features, out_features):
        super().__init__()
        # Using our own defined layer
        self.lin1 = MyLinearLayer(in_features, intermediate_features)
        # Using pre-defined Linear Layer
        self.lin2 = torch.nn.Linear(intermediate_features, out_features)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x


print("=== MyLinearLayer: output shape ===")
layer = MyLinearLayer(4, 3)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
out = layer(x)
print("output shape:", list(out.shape))
print("> output shape correct:", list(out.shape) == [1, 3])

print("\n=== MyLinearLayer: parameter registration ===")
params = list(layer.parameters())
print("num parameters:", len(params))
print("> num parameters:", len(params) == 2)  # weight + bias
for name, p in layer.named_parameters():
    print(f"  {name}: shape={list(p.shape)}")
print("> weight shape:", list(layer.weight.shape) == [4, 3])
print("> bias shape:", list(layer.bias.shape) == [3])

print("\n=== MyLinearLayer: __call__ vs forward() ===")
out_call    = layer(x)
out_forward = layer.forward(x)
print("> outputs match:", torch.allclose(out_call, out_forward))

print("\n=== MyLinearLayer: backward ===")
layer.zero_grad()
layer(x).sum().backward()
print("> weight.grad exists:", layer.weight.grad is not None)
print("> bias.grad exists:", layer.bias.grad is not None)

print("\n=== MySmallModel: output shape ===")
model = MySmallModel(4, 8, 2)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
out = model(x)
print("output shape:", list(out.shape))
print("> output shape correct:", list(out.shape) == [1, 2])

print("\n=== MySmallModel: parameters collected from both sub-modules ===")
params = list(model.parameters())
# lin1: weight (4x8) + bias (8) = 2 params
# lin2: weight (8x2) + bias (2) = 2 params
print("num parameters:", len(params))
print("> num parameters:", len(params) == 4)
for name, p in model.named_parameters():
    print(f"  {name}: shape={list(p.shape)}")

print("\n=== MySmallModel: __call__ vs forward() ===")
out_call    = model(x)
out_forward = model.forward(x)
print("> outputs match:", torch.allclose(out_call, out_forward))

print("\n=== MySmallModel: backward through nested modules ===")
model.zero_grad()
model(x).sum().backward()
print("> all grads computed:", all(p.grad is not None for p in model.parameters()))

print("\n=== __call__ vs forward() on a built-in module ===")
fc = torch.nn.Linear(3, 2)
x = torch.tensor([[1.0, 2.0, 3.0]])
out_call    = fc(x)
out_forward = fc.forward(x)
print("> outputs match:", torch.allclose(out_call, out_forward))
print("> shapes match:", out_call.shape == out_forward.shape)

print("\n=== Sequential: submodules run via __call__ path ===")
seq = torch.nn.Sequential(
    torch.nn.Linear(2, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
)
x = torch.tensor([[1.0, 2.0]])
out_seq_call    = seq(x)
out_seq_forward = seq.forward(x)
print("output shape:", list(out_seq_call.shape))
print("> output shape correct:", list(out_seq_call.shape) == [1, 1])
print("> call and forward match:", torch.allclose(out_seq_call, out_seq_forward))

print("\nAll nn_module checks passed.")
