import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[List[int], int, torch.Size], eps=1e-05, elementwise_affine=True,
                 bias = True):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        if isinstance(normalized_shape, List):
            normalized_shape = torch.Size(normalized_shape)

        self.normalized_shape = normalized_shape
        self.dimensions = tuple(range(-len(self.normalized_shape), 0))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        

        if self.elementwise_affine:
            self.weight = nn.parameter.Parameter(torch.ones(normalized_shape))
            if bias:
                self.bias = nn.parameter.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=self.dimensions, keepdim=True)
        var = x.var(dim=self.dimensions, keepdim=True, unbiased=False)
        out = (x - mean)/torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            out = out * self.weight
            if self.bias is not None:
                out = out + self.bias
        return out
     
class MyRMSNorm(nn.Module):
    def __init__(self, normalized_shape: Union[List[int], int, torch.Size], eps=None,
                 elementwise_affine: bool=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        if isinstance(normalized_shape, List):
            normalized_shape = torch.Size(normalized_shape)
        self.normalized_shape = normalized_shape

        self.n = 1
        for i in range(len(self.normalized_shape)):
            self.n *= self.normalized_shape[i]

        self.dimension = tuple(range(-len(self.normalized_shape), 0))
        self.elementwise_affine = elementwise_affine
        if eps is None:
            self.eps = torch.finfo(torch.float64).eps
        else:
            self.eps = eps
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor):
        sum = torch.square(x).sum(dim=self.dimension, keepdim=True)
        rms = torch.sqrt((sum / self.n) + self.eps)
        print(rms)
        out = x / rms
        if self.weight is not None:
            out = out * self.weight
        return out