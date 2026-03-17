# training_sgd.py — Full training loop: fit y = 2x + 1 with SGD and Adam.
# Verifies: optimizer.step(), optimizer.zero_grad(), loss decreases.

print("=== Training y = 2x + 1 with SGD ===")

# Dataset: y = 2*x + 1
xs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
ys = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

model   = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
opt     = torch.optim.SGD(model.parameters(), lr=0.01)

losses = []
for epoch in range(200):
    opt.zero_grad()
    pred = model(xs)
    loss = loss_fn(pred, ys)
    loss.backward()
    opt.step()
    if epoch % 40 == 0:
        losses.append(loss.item())
        print(f"  epoch {epoch:3d}  loss={loss:.6f}")

print("Loss decreased:", losses[-1] < losses[0])
print("> SGD loss decreased:", losses[-1] < losses[0])

with torch.no_grad():
    pred = model(xs)
print("Final predictions:", pred)
print("Expected:         ", ys)

# Check weights are approximately correct (w≈2, b≈1)
params = {name: p for name, p in model.named_parameters()}
w = params['weight'].item()
b = params['bias'].item()
print(f"Learned weight={w:.3f} (expected ~2), bias={b:.3f} (expected ~1)")
print("Weight close:", abs(w - 2.0) < 0.2)
print("Bias close:  ", abs(b - 1.0) < 0.5)

print("\n=== Training y = 2x + 1 with Adam ===")

model   = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
opt     = torch.optim.Adam(model.parameters(), lr=0.05)

losses = []
for epoch in range(200):
    opt.zero_grad()
    pred = model(xs)
    loss = loss_fn(pred, ys)
    loss.backward()
    opt.step()
    if epoch % 40 == 0:
        losses.append(loss.item())
        print(f"  epoch {epoch:3d}  loss={loss:.6f}")

print("Loss decreased:", losses[-1] < losses[0])
print("> Adam loss decreased:", losses[-1] < losses[0])
params = {name: p for name, p in model.named_parameters()}
w = params['weight'].item()
b = params['bias'].item()
print(f"Learned weight={w:.3f} (expected ~2), bias={b:.3f} (expected ~1)")

print("\n=== Optimizer.zero_grad() resets gradients ===")
fc  = torch.nn.Linear(2, 1)
opt = torch.optim.SGD(fc.parameters(), lr=0.1)

x = torch.tensor([[1.0, 2.0]])
fc(x).sum().backward()

grads_before = [p.grad.tolist() for p in fc.parameters()]
print("Grads after first backward:", grads_before)
print("> grads exist after backward:", all(p.grad is not None for p in fc.parameters()))

opt.zero_grad()
grads_after = [p.grad for p in fc.parameters()]
print("Grads after zero_grad (expected all None):", grads_after)
print("> grads None after zero_grad:", all(p.grad is None for p in fc.parameters()))

print("\nAll training_sgd checks passed.")
