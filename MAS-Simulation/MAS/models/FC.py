"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: A FC model for device state prediction.
Date: 2025-05-14
"""

from typing import Any, Dict, Tuple, override

import torch
import torch.nn as nn

from .modelBase import ModelBase


class ModelFC(ModelBase):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(ModelFC, self).__init__(input_size, output_size, hidden_size, num_layers)

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            *[layer for _ in range(num_layers) for layer in (nn.Linear(hidden_size, hidden_size), nn.LeakyReLU())],
            nn.Linear(hidden_size, 1)
        )

    @override
    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.fc.forward(input)
        return (out, torch.empty(1))
