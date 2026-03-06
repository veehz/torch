import torch

x = torch.nn.Linear(2, 3)
with torch.no_grad():
    x.weight.copy_(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32))
    x.bias.copy_(torch.tensor([1, 2, 3], dtype=torch.float32))

input = torch.tensor([1, 2], dtype=torch.float32)
output = x(input).sum()
output.backward()
print(x.weight.grad)
print(x.bias.grad)