"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: Base class for neural network models that defines a common interface.
Date: 2025-05-14
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class ModelBase(ABC, nn.Module):
    def get_hparams(self) -> Dict[str, Any]:
        return {}
    
    @abstractmethod
    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass