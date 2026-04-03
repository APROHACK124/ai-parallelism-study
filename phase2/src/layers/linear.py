import torch
class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        y = x @ self.weight.T
        
        if self.bias is not None:
            y = y + self.bias
        
        return y