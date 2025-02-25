# Imports

from torch import nn
from torch.nn import functional as F
from torch.nn import Module

########################################################################################


class DefaultGenerator(Module):
    def __init__(self, N_i=16, N_o=2, **kwargs):
        super().__init__()

        self.dense = nn.Linear(N_i, N_o)
        pass

    def forward(self, x):
        y = self.dense(x)
        return y

    pass
