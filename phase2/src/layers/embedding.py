import torch

class MyEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        if padding_idx is not None:
            if not (-num_embeddings <= padding_idx < num_embeddings):
                raise ValueError("padding_idx out of range")
            if padding_idx < 0:
                padding_idx += num_embeddings
            self.padding_idx = padding_idx

        self.weight = torch.nn.parameter.Parameter(torch.zeros((num_embeddings, embedding_dim)))
        self.reset_parameters()

        if self.padding_idx is not None:
            self.weight.register_hook(self._mask_padding_grad)

    def _mask_padding_grad(self, grad: torch.Tensor):
        grad = grad.clone()
        grad[self.padding_idx].zero_()
        return grad

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].zero_()

    def forward(self, x):
        return self.weight[x]
        

