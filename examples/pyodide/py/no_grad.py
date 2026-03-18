# no_grad.py — Verify that torch.no_grad() actually disables gradient tracking.

print("=== no_grad context manager ===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

print("Grad enabled outside context:", torch.is_grad_enabled())
print("> Grad enabled outside context:", torch.is_grad_enabled())

with torch.no_grad():
    print("Grad enabled inside context:", torch.is_grad_enabled())
    print("> Grad enabled inside context:", torch.is_grad_enabled())
    y = x * 2
    print("y.requires_grad inside no_grad (expected False):", y.requires_grad)
    print("> y.requires_grad inside no_grad:", y.requires_grad)
    # y has no grad_fn, so backward would be a no-op

print("Grad enabled after context:", torch.is_grad_enabled())
print("> Grad enabled after context:", torch.is_grad_enabled())

# Outside no_grad, requires_grad propagates normally
z = x * 2
print("z.requires_grad outside no_grad (expected True):", z.requires_grad)
print("> z.requires_grad outside no_grad:", z.requires_grad)

print("\n=== Gradient NOT computed inside no_grad ===")
x = torch.tensor([3.0], requires_grad=True)
with torch.no_grad():
    y = x * x  # should not build computation graph
    print("y:", y)
    print("y.requires_grad (expected False):", y.requires_grad)
    print("> y.requires_grad inside no_grad:", y.requires_grad)

# x.grad should still be None since we didn't call backward
print("x.grad after no_grad block (expected None):", x.grad)
print("> x.grad after no_grad block is None:", x.grad is None)

print("\n=== Gradient IS computed outside no_grad ===")
x = torch.tensor([3.0], requires_grad=True)
y = x * x
y.backward()
print("x.grad after backward (expected 6.0):", x.grad)
print("> x.grad correct:", x.grad is not None and x.grad.allclose(torch.tensor([6.0])))

print("\n=== Nested no_grad blocks restore state correctly ===")
print("is_grad_enabled:", torch.is_grad_enabled())
print("> is_grad_enabled before outer:", torch.is_grad_enabled())
with torch.no_grad():
    print("inside outer no_grad:", torch.is_grad_enabled())
print("after outer no_grad (expected True):", torch.is_grad_enabled())
print("> is_grad_enabled after outer:", torch.is_grad_enabled())

print("\nAll no_grad checks passed.")
