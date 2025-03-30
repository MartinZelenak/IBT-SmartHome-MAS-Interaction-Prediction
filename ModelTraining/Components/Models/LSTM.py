from typing import Any, Dict, List, Tuple, override

import torch
import torch.nn as nn

from ..ModelBase import ModelBase


class PerDeviceLSTM(ModelBase):
    def __init__(self, shared_input_size, per_device_input_size, n_devices, hidden_size, num_layers):
        super(PerDeviceLSTM, self).__init__()

        self.shared_input_size = shared_input_size
        self.per_device_input_size = per_device_input_size
        self.n_devices = n_devices
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.device_models = nn.ModuleList([
            nn.ModuleList([
                nn.LSTM(shared_input_size + per_device_input_size, hidden_size, num_layers, batch_first=True),
                nn.Linear(hidden_size, 1)
            ]) for _ in range(n_devices)
        ])

    @override
    def get_hparams(self) -> Dict[str, Any]:
        return {
            "shared_input_size": self.shared_input_size,
            "per_device_input_size": self.per_device_input_size,
            "n_devices": self.n_devices,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers
        }

    @override
    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forwards the given input through the model for each device

        Args:
            input (torch.Tensor): Input with dimensions [batch_size, sequence_len, shared_input_size + (per_device_input_size * n_devices)]
            hidden_state (torch.Tensor | None): Hidden state with dimensions [n_devices, 2, num_layers, batch_size, ...] or None

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output with dimensions [batch_size, sequence_len, n_devices] and new hidden state with dimensions [n_devices, 2, num_layers, batch_size, ...]
        """
        batch_size, sequence_len, _ = input.size()
        output = torch.zeros((batch_size, sequence_len, self.n_devices), device=input.device)

        if hidden_state is not None and hidden_state.size(3) > batch_size:
            hidden_state = hidden_state[:, :, :, :batch_size, :].contiguous()

        out_hidden_states: List[torch.Tensor] = []

        for i in range(self.n_devices):
            # Take the shared and device specific parts of the input
            device_input = torch.cat(
                [
                    input[:, :, : self.shared_input_size],
                    input[:, :, self.shared_input_size + i : self.shared_input_size + i + self.per_device_input_size],
                ],
                dim=2,
            )

            # LSTM
            device_output, out_hidden_state = self.device_models[i][0](     # type: ignore
                device_input, 
                (hidden_state[i, 0], hidden_state[i, 1]) if hidden_state is not None else None
            )
            ## stack h_n and c_n
            out_hidden_state = torch.stack(out_hidden_state, dim=0)
            out_hidden_states.append(out_hidden_state)

            # Linear
            device_output = self.device_models[i][1](device_output)     # type: ignore
            output[:, :, i] = device_output.squeeze(-1)

        return output, torch.stack(out_hidden_states, dim=0)
