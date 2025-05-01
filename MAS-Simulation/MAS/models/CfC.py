from typing import Any, Dict, Tuple, override

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

from .modelBase import ModelBase

class ModelCfC(ModelBase):

    def __init__(self, input_size, output_size, num_neurons, _=1):
        """_ (int, optional): Throw away parameter to conform to the ModelBase interface. Doesn't do anything. Defaults to 1.
        """
        super(ModelCfC, self).__init__(input_size, output_size, num_neurons, 1)

        self.num_neurons = num_neurons

        self.cfc = CfC(input_size, AutoNCP(self.num_neurons, output_size), batch_first=True)

    @override
    def get_hparams(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "num_neurons": self.num_neurons,
        }

    @override
    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hx = self.cfc.forward(input, hidden_state)

        return out, hx
