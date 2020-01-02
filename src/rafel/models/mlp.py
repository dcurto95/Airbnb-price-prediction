import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.hidden1 = nn.Linear(n_input_features, 100)
        self.hidden2 = nn.Linear(100, 50)
        self.hidden3 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 1)

        # Activations
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, inp):
        x = self.hidden1(inp)
        x = self.leaky_relu(x)
        x = self.hidden2(x)
        x = self.leaky_relu(x)
        x = self.hidden3(x)
        x = self.leaky_relu(x)
        x = self.out(x)
        x = self.tanh(x)
        return x
