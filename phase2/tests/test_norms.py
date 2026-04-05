import torch
from src.layers.norms import MyLayerNorm
from torch.testing import assert_close

N ,H, W = 5, 3, 2

# Here I will apply normalization over the last two dimensions of the tensor
real_input = torch.randn(N, H, W, requires_grad=True)
my_input = real_input.detach().clone().requires_grad_()

real_layer_norm = torch.nn.LayerNorm([3, 2])
my_layer_norm = MyLayerNorm([3, 2])

assert_close(real_layer_norm.weight, my_layer_norm.weight)
assert_close(real_layer_norm.bias, my_layer_norm.bias)

real_output = real_layer_norm(real_input)
my_output = my_layer_norm(my_input)

# print(real_output, my_output)
assert_close(real_output, my_output)

real_output.sum().backward()
my_output.sum().backward()

assert_close(my_layer_norm.weight.grad, real_layer_norm.weight.grad)
assert_close(my_layer_norm.bias.grad, real_layer_norm.bias.grad)
assert_close(real_input.grad, my_input.grad)

# Checking state_dict functionality

copy_layerNorm = MyLayerNorm(real_input[0].shape)
copy_layerNorm.load_state_dict(my_layer_norm.state_dict())

x = torch.randn_like(my_input)
x_cp = x.detach().clone()
assert_close(copy_layerNorm(x_cp), my_layer_norm(x))

# Checkig with elementwise_affine = False

my_layer_norm = MyLayerNorm(2, eps=0.001, elementwise_affine=False)
real_layer_norm = torch.nn.LayerNorm(2, eps=0.001, elementwise_affine=False)

my_input = torch.randn_like(my_input)
real_input = my_input.detach().clone()

assert_close(my_layer_norm(my_input), real_layer_norm(real_input))

print("ok!")

# other forms of normalized_shape also work!

# Another gradient check recommended by AI

from torch.autograd import gradcheck

m = MyLayerNorm(4).double()

x = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)

gradcheck(lambda inp: m(inp), (x, ))


### Here starts the testing for RMSNorm
from src.layers.norms import MyRMSNorm
import torch
from torch.testing import assert_close

my_x = torch.randn(4, 2, 3, requires_grad=True)
real_x = my_x.detach().clone().requires_grad_()

my_RMSNorm = MyRMSNorm([2, 3], elementwise_affine=False)
real_RMSNorm = torch.nn.RMSNorm([2, 3], elementwise_affine=False)

assert_close(real_RMSNorm(real_x), my_RMSNorm(my_x))

my_out = my_RMSNorm(my_x).sum()
real_out = real_RMSNorm(real_x).sum()

my_out.backward()
real_out.backward()

assert_close(my_x.grad, real_x.grad)

# So far so good, let's try with elementwise_affine=True

my_RMSNorm = MyRMSNorm([3], elementwise_affine=True)
real_RMSNorm = torch.nn.RMSNorm([3], elementwise_affine=True)

assert_close(my_RMSNorm.weight, real_RMSNorm.weight)

my_x = torch.randn([3, 3], requires_grad=True)
real_x = my_x.detach().clone().requires_grad_()

my_out = my_RMSNorm(my_x).mean()
real_out = real_RMSNorm(real_x).mean()

assert_close(my_out, real_out)
my_out.backward()
real_out.backward()
assert_close(my_x.grad, real_x.grad)
assert_close(my_RMSNorm.weight.grad, real_RMSNorm.weight.grad)


# State dict testing

cp_norm = MyRMSNorm([3], elementwise_affine=True)
cp_norm.load_state_dict(my_RMSNorm.state_dict())

assert_close(cp_norm(my_x), my_RMSNorm(my_x))
my_RMSNorm(my_x)
