from src.layers.linear import *
import torch
from torch.testing import assert_close

torch.manual_seed(0)

mine = Linear(5, 7, bias=True)
ref = torch.nn.Linear(5, 7, bias=True)

with torch.no_grad():
    ref.weight.copy_(mine.weight)
    ref.bias.copy_(mine.bias)

x = torch.randn(2, 3, 5, requires_grad=True)
x_ref = x.detach().clone().requires_grad_(True)

y = mine(x)
y_ref = ref(x_ref)
assert_close(y, y_ref)

loss = y.square().mean()
loss_ref = y_ref.square().mean()

loss.backward()
loss_ref.backward()

assert_close(mine.weight.grad, ref.weight.grad)
assert_close(mine.bias.grad, ref.bias.grad)
assert_close(x.grad, x_ref.grad)

clone = Linear(5, 7, bias=True)
clone.load_state_dict(mine.state_dict())
assert_close(clone(x.detach()), mine(x.detach()))