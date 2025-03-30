from typing import Any, Dict, Tuple, override

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

from ..ModelBase import ModelBase


class PerDeviceCfC(ModelBase):
    def __init__(self, shared_input_size: int, per_device_input_size: int, n_devices: int, n_neurons: int):
        super(PerDeviceCfC, self).__init__()

        self.shared_input_size = shared_input_size
        self.per_device_input_size = per_device_input_size
        self.n_devices = n_devices
        self.n_neurons = n_neurons

        self.device_models = nn.ModuleList(
            [
                CfC(shared_input_size + per_device_input_size, AutoNCP(n_neurons, 1), batch_first=True)
                for _ in range(n_devices)
            ]
        )

    @override
    def get_hparams(self) -> Dict[str, Any]:
        return {
            "shared_input_size": self.shared_input_size,
            "per_device_input_size": self.per_device_input_size,
            "n_devices": self.n_devices,
            "n_neurons": self.n_neurons
        }

    @override
    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forwards the given input through a CfC model for each device

        Args:
            input (torch.Tensor): Input with dimensions [batch_size, sequence_len, shared_input_size + (per_device_input_size * n_devices)]
            hidden_state (torch.Tensor | None): Hidden state with dimensions [n_devices, batch_size, n_neurons] or None

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output with dimensions [batch_size, sequence_len, n_devices] and new hidden state with dimensions [n_devices, batch_size, n_neurons]
        """
        batch_size, sequence_len, _ = input.size()
        output = torch.zeros((batch_size, sequence_len, self.n_devices), device=input.device)

        if hidden_state is None:
            hidden_state = torch.zeros((self.n_devices, batch_size, self.n_neurons), device=input.device)
        elif hidden_state.size(1) > batch_size:
            hidden_state = hidden_state[:, : batch_size, :]

        out_hidden_state = torch.zeros_like(hidden_state)

        for i in range(self.n_devices):
            device_input = torch.cat(
                [
                    input[:, :, : self.shared_input_size],
                    input[:, :, self.shared_input_size + i : self.shared_input_size + i + self.per_device_input_size],
                ],
                dim=2,
            )
            device_output, out_hidden_state[i] = self.device_models[i](device_input, hidden_state[i])
            output[:, :, i] = device_output.squeeze(-1)

        return output, out_hidden_state