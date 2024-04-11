import torch


class OutputRateCoding(torch.nn.Module):
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        x = torch.stack(x, dim=0)
        return x.mean(dim=self.dim)