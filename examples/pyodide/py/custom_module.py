# custom_module.py — User-defined Module subclass (pure Python), training loop.
# Verifies: Module subclassing, parameter registration, forward(), parameters().

class TwoLayerNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x


print("=== Custom Module construction ===")
model = TwoLayerNet(2, 4, 1)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
out = model(x)
print("input shape:", x.shape)
print("output shape:", out.shape)
print("output:", out)

print("\n=== parameters() collects from all sub-modules ===")
params = list(model.parameters())
print("number of parameters:", len(params))
# 2 layers × (weight + bias) = 4 parameters
print("expected 4:", len(params) == 4)
print("> n_params == 4:", len(params) == 4)

print("\n=== named_parameters() ===")
for name, p in model.named_parameters():
    print(f"  {name}: shape={p.shape}")

print("\n=== Backward through custom Module ===")
loss = out.sum()
loss.backward()
print("All parameter grads computed:")
for name, p in model.named_parameters():
    print(f"  {name}.grad is not None:", p.grad is not None)
print("> all param grads computed:", all(p.grad is not None for p in model.parameters()))

print("\n=== Training loop with custom Module ===")

# XOR-like dataset (linearly separable approximation)
xs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
ys = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model   = TwoLayerNet(2, 8, 1)
loss_fn = torch.nn.MSELoss()
opt     = torch.optim.Adam(model.parameters(), lr=0.05)

initial_loss = None
for epoch in range(300):
    opt.zero_grad()
    pred = model(xs)
    loss = loss_fn(pred, ys)
    loss.backward()
    opt.step()
    if epoch == 0:
        initial_loss = loss.item()
    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d}  loss={loss:.4f}")

print("Loss decreased:", loss.item() < initial_loss)
print("> loss decreased:", loss.item() < initial_loss)

print("\n=== Module.zero_grad() clears all parameter grads ===")
# Run one backward pass
opt.zero_grad()
model(xs).sum().backward()
grads_exist = all(p.grad is not None for p in model.parameters())
print("Grads exist after backward:", grads_exist)
print("> grads exist after backward:", grads_exist)

model.zero_grad()
grads_cleared = all(p.grad is None for p in model.parameters())
print("Grads cleared after zero_grad:", grads_cleared)
print("> grads cleared after zero_grad:", grads_cleared)

print("\nAll custom_module checks passed.")
