import torch
from src.layers.embedding import MyEmbedding
from torch.testing import assert_close

# Test withoug padding_idx

my_emb = MyEmbedding(5, 3)
real_emb = torch.nn.Embedding(5, 3)

with torch.no_grad():
    real_emb.weight.copy_(my_emb.weight)

my_x = torch.LongTensor([[0, 1], [1, 2], [2, 1]])
real_x = my_x.detach().clone()

my_y = my_emb(my_x)
real_y = real_emb(real_x)

assert_close(my_y, real_y)
print(f"Result my embedding: {my_y}, result real embedding: {real_y}")

my_y.sum().backward()
real_y.sum().backward()

assert_close(my_emb(my_x), real_emb(real_x))
assert_close(my_emb.weight.grad, real_emb.weight.grad)
print(f'gradient my embedding: {my_emb.weight.grad}, gradient real_emb: {real_emb.weight.grad}')

# This is consistent with the expected value! 

# Now let's test with the padding_idx attribute

# (writting from scratch to check if I learned the structure)


import torch
from src.layers.embedding import MyEmbedding
from torch.testing import assert_close

my_emb = MyEmbedding(4, 3, padding_idx=1)
real_emb = torch.nn.Embedding(4, 3, padding_idx=1)


# Uncomment to change weight at padding_idx, which is allowed in pytorch's nn.Embedding
# with torch.no_grad():
#     my_emb.weight[1].copy_(torch.tensor([1.0, 2.0, 3.0]))

with torch.no_grad():
    real_emb.weight.copy_(my_emb.weight)

my_x = torch.tensor([2, 1, 0])
real_x = my_x.detach().clone()

my_y = my_emb(my_x)
real_y = real_emb(real_x)

my_y.mean().backward()
real_y.mean().backward()

assert_close(my_y, real_y)
assert_close(my_emb.weight.grad, real_emb.weight.grad)

print(my_emb.weight.grad, real_emb.weight.grad)

print("We can see that the gradient at padding_idx remains 0 !")

print(my_emb(torch.tensor([1])), real_emb(torch.tensor([1])))


# state dict testing
my_emb = MyEmbedding(4, 3, padding_idx=1)
cp = MyEmbedding(4, 3, padding_idx=1)
cp.load_state_dict(my_emb.state_dict())

assert_close(cp.weight, my_emb.weight)

my_x = torch.tensor([1, 1, 2, 3])
cp_x = my_x.detach().clone()
assert_close(my_emb(my_x), cp(cp_x))