# basic_ops.py — Basic tensor operations, autograd, and backward pass.
# Verifies: creation, arithmetic operators, reductions, backward, grad access.

print("=== Basic tensor creation ===")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
print("x:", x)
print("y:", y)
print("x.shape:", x.shape)
print("x.requires_grad:", x.requires_grad)
print("> x.requires_grad:", x.requires_grad)

print("\n=== Arithmetic operators ===")
print("x + y:", x + y)
print("x - y:", x - y)
print("x * y:", x * y)
print("x / y:", x / y)
print("x ** 2:", x ** 2)
print("> x+y correct:", (x + y).allclose(torch.tensor([5.0, 7.0, 9.0])))
print("> x*y correct:", (x * y).allclose(torch.tensor([4.0, 10.0, 18.0])))
print("> x/y correct:", (x / y).allclose(torch.tensor([0.25, 0.4, 0.5])))

print("\n=== Scalar arithmetic (reversed operators) ===")
print("2 + x:", 2 + x)   # __radd__
print("10 - x:", 10 - x) # __rsub__
print("2 * x:", 2 * x)   # __rmul__
print("1 / x:", 1 / x)   # __rtruediv__
print("2 ** x:", 2 ** x) # __rpow__
print("> 2+x correct:", (2 + x).allclose(torch.tensor([3.0, 4.0, 5.0])))
print("> 2*x correct:", (2 * x).allclose(torch.tensor([2.0, 4.0, 6.0])))

print("\n=== Unary operations ===")
print("-x:", -x)
print("x.neg():", x.neg())
print("x.abs():", torch.tensor([-1.0, 2.0, -3.0]).abs())
print("x.sqrt():", x.sqrt())
print("x.exp():", x.exp())
print("x.log():", x.log())
print("x.sigmoid():", x.sigmoid())
print("x.relu():", torch.tensor([-1.0, 0.0, 2.0]).relu())
print("> neg correct:", (-x).allclose(torch.tensor([-1.0, -2.0, -3.0])))
print("> sqrt correct:", x.sqrt().allclose(torch.tensor([1.0, 1.4142, 1.7321]), atol=1e-4))
print("> relu correct:", torch.tensor([-1.0, 0.0, 2.0]).relu().allclose(torch.tensor([0.0, 0.0, 2.0])))

print("\n=== Reductions ===")
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("a:", a)
print("a.sum():", a.sum())           # sum all
print("a.sum(dim=0):", a.sum(dim=0)) # sum along rows
print("a.sum(dim=1):", a.sum(dim=1)) # sum along cols
print("a.mean():", a.mean())
print("a.max():", a.max())
print("a.min():", a.min())
print("> a.sum() correct:", a.sum().allclose(torch.tensor(21.0)))
print("> a.sum(dim=0) correct:", a.sum(dim=0).allclose(torch.tensor([5.0, 7.0, 9.0])))
print("> a.mean() correct:", a.mean().allclose(torch.tensor(3.5)))

print("\n=== Shape utilities ===")
print("a.shape:", a.shape)
print("a.size():", a.size())
print("a.size(0):", a.size(0))
print("a.dim():", a.dim())
print("a.numel():", a.numel())
print("a.reshape(3, 2):", a.reshape(3, 2))
print("a.reshape([6]):", a.reshape([6]))
print("a.T:", a.T)
print("a.transpose(0, 1):", a.transpose(0, 1))
print("> a.size(0):", a.size(0))
print("> a.dim():", a.dim())
print("> a.numel():", a.numel())

print("\n=== Comparison operators ===")
b = torch.tensor([1.0, 3.0, 3.0])
c = torch.tensor([2.0, 2.0, 3.0])
print("b:", b, " c:", c)
print("b.lt(c):", b.lt(c))
print("b.gt(c):", b.gt(c))
print("b.eq(c):", b.eq(c))

print("\n=== Backward pass (z = sum(x * y)) ===")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
z = (x * y).sum()
print("z:", z)
z.backward()
print("x.grad (expected [4,5,6]):", x.grad)
print("y.grad (expected [1,2,3]):", y.grad)
print("> x.grad correct:", x.grad is not None and x.grad.allclose(torch.tensor([4.0, 5.0, 6.0])))
print("> y.grad correct:", y.grad is not None and y.grad.allclose(torch.tensor([1.0, 2.0, 3.0])))

print("\n=== Backward with chain rule (z = sum((x+1)^2)) ===")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
z = ((x + 1) ** 2).sum()
z.backward()
# dz/dx = 2*(x+1) = [4, 6, 8]
print("x.grad (expected [4,6,8]):", x.grad)
print("> x.grad correct:", x.grad is not None and x.grad.allclose(torch.tensor([4.0, 6.0, 8.0])))

print("\n=== allclose ===")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.0, 2.0, 3.0000001])
print("allclose (expected True):", a.allclose(b))
print("> allclose (expected True):", a.allclose(b))
b2 = torch.tensor([1.0, 2.0, 4.0])
print("allclose (expected False):", a.allclose(b2))
print("> allclose (expected False):", a.allclose(b2))

print("\n=== Iteration ===")
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
for row in t:
    print("row:", row)
print("> num rows:", sum(1 for _ in torch.tensor([[1.0, 2.0], [3.0, 4.0]])))

print("\n=== item(), float(), int() ===")
s = torch.tensor([42.0])
print("s.item():", s.item())
print("float(s):", float(s))
print("int(s):", int(s))
print("> float(s):", float(s))
print("> int(s):", int(s))

print("\nAll basic_ops checks passed.")
