from typing import Any, Dict, Tuple, override

import torch
import torch.nn as nn

from .modelBase import ModelBase

class ModelLSTM(ModelBase):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(ModelLSTM, self).__init__(input_size, output_size, hidden_size, num_layers)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    @override
    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hx = None
        if hidden_state is not None:
            hx = (hidden_state[0], hidden_state[1])

        out, (h, c) = self.lstm.forward(input, hx)
        out = self.fc.forward(out)

        # Expand hx to match the input dimensionality
        if input.dim() > h.dim():
            hx_dim_diff = input.dim() - h.dim()
            for _ in range(hx_dim_diff):
                h = h.unsqueeze(0)
                c = c.unsqueeze(0)

        hx = torch.stack((h, c), dim=0)
        return out, hx
