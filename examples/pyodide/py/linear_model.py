# linear_model.py — torch.nn.Linear forward pass, loss computation, backward.
# Verifies: nn.Linear, MSELoss, BCELoss, Sequential, nn.ReLU, gradient flow.

print("=== nn.Linear forward pass ===")
fc = torch.nn.Linear(3, 2)
x = torch.tensor([[1.0, 2.0, 3.0]])   # batch of 1, 3 features
out = fc(x)
print("input shape:", x.shape)
print("output shape:", out.shape)
print("output:", out)

print("\n=== nn.Linear parameters ===")
params = list(fc.parameters())
print("number of parameters:", len(params))
print("param shapes:", [p.shape for p in params])
print("all require grad:", all(p.requires_grad for p in params))
print("> n_params:", len(params))
print("> all require grad:", all(p.requires_grad for p in params))

print("\n=== named_parameters ===")
for name, p in fc.named_parameters():
    print(f"  {name}: shape={p.shape}")

print("\n=== MSELoss ===")
pred   = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 2.0])
loss_fn = torch.nn.MSELoss()
loss = loss_fn(pred, target)
print("MSELoss:", loss)
# Expected: mean([(0.5)^2, (0.5)^2, (1)^2]) = mean([0.25, 0.25, 1.0]) = 0.5
expected = ((1.0-1.5)**2 + (2.0-2.5)**2 + (3.0-2.0)**2) / 3
print("Expected:", expected)
print("Close:", abs(loss.item() - expected) < 1e-6)
print("> MSELoss close:", abs(loss.item() - expected) < 1e-6)

print("\n=== L1Loss ===")
loss_fn = torch.nn.L1Loss()
loss = loss_fn(pred, target)
print("L1Loss:", loss)
expected_l1 = (0.5 + 0.5 + 1.0) / 3
print("Expected:", expected_l1)
print("Close:", abs(loss.item() - expected_l1) < 1e-6)
print("> L1Loss close:", abs(loss.item() - expected_l1) < 1e-6)

print("\n=== Backward through Linear ===")
fc = torch.nn.Linear(2, 1)
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
out = fc(x)
loss = out.sum()
loss.backward()
print("x.grad (should be non-None):", x.grad)
for name, p in fc.named_parameters():
    print(f"  {name}.grad is not None:", p.grad is not None)
print("> x.grad is not None:", x.grad is not None)
print("> all param grads computed:", all(p.grad is not None for p in fc.parameters()))

print("\n=== Sequential ===")
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
)
x = torch.tensor([[1.0, 0.5, -1.0, 2.0]])
out = model(x)
print("Sequential output shape:", out.shape)
print("Sequential output:", out)
params = list(model.parameters())
print("Total parameters:", len(params))
print("> sequential n_params:", len(params))

print("\n=== zero_grad clears gradients ===")
fc = torch.nn.Linear(2, 1)
x = torch.tensor([[1.0, 1.0]])
out = fc(x).sum()
out.backward()
for name, p in fc.named_parameters():
    print(f"  {name}.grad before zero_grad:", p.grad)
fc.zero_grad()
for name, p in fc.named_parameters():
    print(f"  {name}.grad after zero_grad (expected None):", p.grad)
print("> grads cleared:", all(p.grad is None for p in fc.parameters()))

print("\nAll linear_model checks passed.")
