from typing import Any, Dict, Tuple, override

import torch
import torch.nn as nn

from ..ModelBase import ModelBase


class PerDeviceFC(ModelBase):
    def __init__(self, shared_input_size, per_device_input_size, n_devices, hidden_size, num_layers):
        super(PerDeviceFC, self).__init__()
        self.shared_input_size = shared_input_size
        self.per_device_input_size = per_device_input_size
        self.n_devices = n_devices
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.device_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_input_size + per_device_input_size, hidden_size),
                nn.LeakyReLU(),
                *[layer for _ in range(num_layers) for layer in (nn.Linear(hidden_size, hidden_size), nn.LeakyReLU())],
                nn.Linear(hidden_size, 1)
            ) for _ in range(n_devices)
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
        batch_size, sequence_len, _ = input.size()
        output = torch.zeros((batch_size, sequence_len, self.n_devices), device=input.device)

        if hidden_state is None:
            hidden_state = torch.zeros(1, device=input.device)

        # Reshape input to merge batch and sequence dimensions
        reshaped_input = input.view(batch_size * sequence_len, -1)
        
        for i in range(self.n_devices):
            # Take the shared and device specific parts of the input
            device_input = torch.cat(
                [
                    reshaped_input[:, : self.shared_input_size],
                    reshaped_input[:, self.shared_input_size + i : self.shared_input_size + i + self.per_device_input_size],
                ],
                dim=1,
            )

            # Make prediction for each of the time step in the input sequence
            device_output = self.device_models[i](device_input)  # type: ignore

            # Reshape output back to original batch and sequence dimensions
            output[:, :, i] = device_output.view(batch_size, sequence_len)

        return (output, hidden_state)

