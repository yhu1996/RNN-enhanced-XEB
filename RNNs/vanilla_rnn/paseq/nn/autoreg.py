# Imports

from abc import abstractmethod, ABC

from typing import Dict, List, Union, Optional, Any, Callable

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module

from .generator import DefaultGenerator


########################################################################################


class AutoregressiveSpinModel(Module, ABC): 
    def __init__(
        self,
        N: int,
        N_i: int = 2,
        N_o: int = 2,
        generator=DefaultGenerator,
        generator_params: Dict[str, Any] = {},
    ):
        super().__init__() 

        self.N = N
        self.N_i = N_i
        self.N_o = N_o
        self.generator = generator(**generator_params)

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = torch.cat(
            [torch.zeros(x.shape[0], 1, x.shape[2]).to(x), x[:, :-1, :]], axis=1
        ) 

        y = self.forward_without_path(y)

        return x, y

    @abstractmethod
    def forward_without_path(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def logprobs(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.forward(x)

        y = (x * y).sum((-2, -1)) 

        return y 

    def sample(self, samples: int) -> torch.Tensor:
        x = torch.zeros(samples, 1, self.N_i).to(next(self.parameters())) 

        for i in range(self.N): 
            y = self.forward_without_path(x)[:, -1:, :] 
            y = F.gumbel_softmax(logits=y, tau=1, hard=True) 
            x = torch.cat([x, y], axis=-2) 

        if self.pathseq is not None:
            x = self.pathseq.inverse(x[:, 1:])
        else:
            x = x[:, 1:]
        

        return x 

    def dataloss(self, x: torch.Tensor) -> torch.Tensor:
        loss = -self.logprobs(x).mean(-1) / self.N   

        return loss


    pass

