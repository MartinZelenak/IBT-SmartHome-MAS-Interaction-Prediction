"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: The base ModelBase class for different nerual netwrok models implementing a common interface.
Date: 2025-05-14
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, TypeVar, Generic

import torch
import torch.nn as nn

T_hidden = TypeVar('T_hidden')

class ModelBase(ABC, nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int):
        super(ModelBase, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def get_hparams(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers
        }
    
    @abstractmethod
    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forwards the given input through the model

        Args:
            input (torch.Tensor): Input with dimensions [sequence_len, input_size]
            hidden_state (T_hidden | None): Hidden state or None

        Returns:
            Tuple[torch.Tensor, T_hidden]: Output with dimensions [sequence_len, output_size] and new hidden state (T_hidden)
        """
        pass