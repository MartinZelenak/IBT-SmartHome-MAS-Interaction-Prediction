"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: A RNN model for device state prediction.
Date: 2025-05-14
"""

from typing import Any, Dict, List, Tuple, override

import torch
import torch.nn as nn

from .modelBase import ModelBase


class ModelRNN(ModelBase):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(ModelRNN, self).__init__(input_size, output_size, hidden_size, num_layers)

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    @override
    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hx = self.rnn.forward(input, hidden_state)
        out = self.fc.forward(out)
        return out, hx