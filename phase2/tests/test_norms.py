import torch
from src.layers.norms import MyLayerNorm
from torch.testing import assert_close

N ,H, W = 5, 3, 2

# Here I will apply normalization over the last two dimensions of the tensor
real_input = torch.randn(N, H, W)
my_input = real_input.detach().clone()

real_layer_norm = torch.nn.LayerNorm(real_input[0].shape)
my_layer_norm = MyLayerNorm(real_input[0].shape)

assert_close(real_layer_norm.weight, my_layer_norm.weight)
assert_close(real_layer_norm.bias, my_layer_norm.bias)

real_output = real_layer_norm(real_input)
my_output = my_layer_norm(my_input)

print(real_output, my_output)


