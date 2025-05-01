import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Any

import torch
import torch.nn as nn

from .systemStats import SystemStats
from .models.modelBase import ModelBase
from .models.FC import ModelFC
from .models.RNN import PerDeviceRNN
from .models.LSTM import ModelLSTM
from .models.CfC import ModelCfC

logger = logging.getLogger("MAS.predictionModel")

MODEL_TYPES = ["CFC", "LSTM", "RNN", "FC"]


@dataclass
class ModelParams:
    type: str
    hidden_size: int
    num_layers: int
    learning_rate: float
    sequence_length: int
    keep_hidden_state: bool

    @staticmethod
    def from_dict(params: dict) -> "ModelParams":
        if str(params["type"]).upper() not in MODEL_TYPES:
            raise ValueError(f"Model type '{params['type']}' is not supported. Supported types: {MODEL_TYPES}")

        return ModelParams(
            type=str(params["type"]),
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            learning_rate=float(params["learning_rate"]),
            sequence_length=int(params["sequence_length"]),
            keep_hidden_state=bool(params["keep_hidden_state"]),
        )


class PredictionModel:
    """A class to represent a prediction model used for learning.
    Parameters:
        input_size (int): The number of input features.
        model_params (ModelParams): Model parameters.
        model_path (Optional[str]): The file path to the trained model to save/load.
        save_after_n_learning_steps (int): The number of learning steps after which to save the model. If <= 0, the model won't be saved.
        collect_stats_as_device (Optional[str]): The device name to collect stats for. If None, no stats will be collected.
    """

    def __init__(
        self,
        input_size: int,
        model_params: ModelParams,
        model_path: Optional[str] = None,
        save_after_n_learning_steps: int = 0,
        collect_stats_for_device: Optional[str] = None,
    ) -> None:
        self.model_params = model_params
        self.save_path = model_path
        self.save_after_n_learning_steps = save_after_n_learning_steps
        self.collect_stats_as_device = collect_stats_for_device
        if model_path and save_after_n_learning_steps > 0:
            self.learn_counter = 0

        # System stats
        if self.collect_stats_as_device:
            SystemStats().predictions[self.collect_stats_as_device] = []
            SystemStats().losses[self.collect_stats_as_device] = []
            SystemStats().actuals[self.collect_stats_as_device] = []

        # Init model
        logger.debug(f"Initializing model with ModelParams: {model_params}")
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model(
            type=self.model_params.type,
            input_size=input_size,
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.num_layers,
        ).to(self.torch_device)
        self.input_sequence = torch.Tensor(size=(0, self.model.input_size)).to(self.torch_device)
        self.input_sequence_len = self.model_params.sequence_length
        self.last_hidden_state: Any | None = None

        self.last_pred: torch.Tensor | None = None
        self.criterion = nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_params.learning_rate)

    def predict(self, state: List[int | float]) -> float:
        """Generate prediction based on current (passed as state) and previous states.
        Passed current state is saved to the sequence.
        """
        if len(state) != self.model.input_size:
            raise ValueError("Number of state features doesn't match the model input size.")

        currentState: torch.Tensor = torch.Tensor(state).to(self.torch_device)
        self.input_sequence = torch.cat((self.input_sequence, currentState.unsqueeze(0)), 0)
        if self.input_sequence.size(0) >= self.input_sequence_len:
            self.input_sequence = self.input_sequence[
                -self.input_sequence_len :
            ]  # keep only last sequence_len samples

        if not self.model_params.keep_hidden_state:
            self.last_hidden_state = None

        pred: torch.Tensor
        pred, hidden_state = self.model.forward(self.input_sequence, self.last_hidden_state)

        pred = self._get_last_tensor_element(pred) # Last prediction in the sequence
        self.last_pred = pred
        if self.model_params.keep_hidden_state:
            self.last_hidden_state = hidden_state.detach()

        pred_value = pred.item()
        if self.collect_stats_as_device:
            SystemStats().predictions[self.collect_stats_as_device].append(pred_value)

        return pred_value

    def learn(self, last_actual: int | float):
        """Learn based on the last prediction and the last actual value.
        The passed last_actual value must correspond to the last predicted value.
        """
        if self.last_pred is None:
            return

        pred = self.last_pred.to(self.torch_device)
        actual = torch.Tensor([float(last_actual)]).to(self.torch_device)

        loss = self.criterion(pred, actual)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_pred = None

        if self.collect_stats_as_device:
            SystemStats().losses[self.collect_stats_as_device].append(loss.item())
            SystemStats().actuals[self.collect_stats_as_device].append(last_actual)

        # Periodic save
        if self.save_path and self.save_after_n_learning_steps > 0:
            self.learn_counter += 1
            if self.learn_counter % self.save_after_n_learning_steps == 0:
                self.learn_counter += 0
                torch.save(self.model.state_dict(), self.save_path)

    def save(self, path: Optional[str] = None) -> bool:
        """Save the model to the specified path.
        If path is None, save_path passed on init will be used instead.
        If path and save_path are both None, the model won't be saved.
        Path should be a string representing the file path.
        """
        save_path = path or self.save_path
        if not save_path:
            return False
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        try:
            torch.save(self.model.state_dict(), save_path)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
        logger.info(f"Model saved to {save_path}")
        return True

    def load(self, path: Optional[str] = None) -> bool:
        """Load the model from the specified path.
        If path is None, save_path passed on init will be used instead.
        If path and save_path are both None, the model won't be loaded.
        Path should be a string representing the file path.
        """
        load_path = path or self.save_path
        if not load_path:
            return False

        if not os.path.exists(load_path):
            return False
        try:
            self.model.load_state_dict(torch.load(load_path))
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
        logger.info(f"Model loaded from {load_path}")
        return True

    def _create_model(self, type: str, input_size: int, hidden_size: int, num_layers: int) -> ModelBase:
        """Creates a model based on the specified type

        Args:
            type (str): Model type (CfC, LSTM, RNN, FC)
            input_size (int): Input size
            hidden_size (int): Hidden size
            num_layers (int): Number of layers

        Returns:
            ModelBase: Created model instance
        """
        if type.upper() == "CFC":
            return ModelCfC(input_size, 1, hidden_size, num_layers)
        elif type.upper() == "LSTM":
            return ModelLSTM(input_size, 1, hidden_size, num_layers)
        elif type.upper() == "RNN":
            return PerDeviceRNN(input_size, 1, hidden_size, num_layers)
        elif type.upper() == "FC":
            return ModelFC(input_size, 1, hidden_size, num_layers)
        else:
            raise ValueError(f"Model type '{type}' is not supported. Supported types: {MODEL_TYPES}")

    def _get_last_tensor_element(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the last number from a tensor

        Args:
            x (torch.Tensor): Input tensor of any dimension

        Returns:
            torch.Tensor: A scalar tensor containing the last element of the input tensor
        """
        while x.dim() > 0:
            x = x[-1]
        return x
