import torch.nn as nn

RELU = 0
LEAKY_RELU = 1
TANH = 2
SIGMOID = 3


class MLP(nn.Module):
    def __init__(self, n_input_features, n_hidden_units=(100, 50, 20, 1), activation_function=LEAKY_RELU):
        super().__init__()

        layers = [nn.Linear(n_input_features, n_hidden_units[0])]
        for i in range(len(n_hidden_units) - 1):
            if activation_function == RELU:
                layers.append(nn.ReLU())
            elif activation_function == LEAKY_RELU:
                layers.append(nn.LeakyReLU())
            elif activation_function == TANH:
                layers.append(nn.Tanh())
            elif activation_function == SIGMOID:
                layers.append(nn.Sigmoid())
            else:
                raise NotImplementedError("Activation function not implemented.")
            layers.append(nn.Linear(n_hidden_units[i], n_hidden_units[i+1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, inp):
        return self.mlp(inp)
