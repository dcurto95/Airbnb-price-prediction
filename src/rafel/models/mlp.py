import torch.nn as nn


class MLP (nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(10, 10, True)

    def forward(self, inp):
        x = self.hidden1(inp)
        return x
