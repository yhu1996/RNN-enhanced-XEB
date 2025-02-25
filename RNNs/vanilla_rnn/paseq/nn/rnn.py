# Imports

from typing import Dict, List, Union, Optional, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

from .autoreg import AutoregressiveSpinModel
from .generator import DefaultGenerator


########################################################################################

# params
## N: number of spins
## N_i: input size
## N_h: hidden size
## N_o: output size

class RNNSpinModel(AutoregressiveSpinModel):
    def __init__(
        self,
        N: int,
        N_i: int = 2,
        N_h: int = 16,
        N_o: int = 2,
        dropout_flag = False,
        dropout_rate = 0.1, 
        generator=DefaultGenerator,
        generator_params: Dict[str, Any] = {},
    ):
        generator_params["N_i"] = N_h 
        generator_params["N_o"] = N_o

        super().__init__(N, N_i, N_o, generator, generator_params)

        self.N_h = N_h
        self.dropout_flag = dropout_flag

        self.gru = nn.GRU(self.N_i, self.N_h, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        pass

    def forward_without_path(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(dtype=torch.float32)
        y = self.gru(y)[0]  
        if self.dropout_flag:
            y = self.dropout(y)
        y = self.generator(y) 
        y = F.log_softmax(y, dim=-1)

        return y

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        pass

    pass

