"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: A PyTorch Dataset classes for sequential data and event-based data,
            including support for data transforms and column removal.
Date: 2025-05-14
"""


from abc import ABC, abstractmethod
from typing import Generator, Optional, Sequence, override

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DataTransform(ABC):
    @abstractmethod
    def __call__(self, sample: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pass

class SequentialStateDataset(Dataset):

    def __init__(
        self, 
        csv_file: str, 
        n_columns: int, 
        max_samples: Optional[int] = None, 
        lookback: int = 1, 
        device: str = "cpu"
    ) -> None:
        data: pd.DataFrame = pd.read_csv(
            csv_file,
            delimiter=",",
            header=0,
            usecols=range(n_columns),
            dtype="float32",
            nrows=max_samples,
            index_col=False,
        )
        self.columns: list[str] = data.columns.tolist()

        self.data: np.ndarray = data.to_numpy()

        self.transforms: Optional[Sequence[DataTransform]] = None
        self.lookback: int = lookback
        self.device = device

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of bounds")

        next_state = self.data[index + self.lookback]
        sequence = self.data[index : index + self.lookback]
        sample = (sequence, next_state)

        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)
                if sample is None:
                    return None

        sample = (
            torch.tensor(sample[0], dtype=torch.float32, device=self.device),
            torch.tensor(sample[1], dtype=torch.float32, device=self.device),
        )

        return sample

    # def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    #     for i in range(len(self)):
    #         sample = self[i]
    #         if sample is not None:
    #             yield sample

    def set_transforms(self, transforms: Optional[Sequence[DataTransform]] = None) -> None:
        self.transforms = transforms

    def sample_sizes(self) -> tuple[tuple[int, int], int]:
        """
        Retrieves the sample sizes from the dataset.

        Returns:
            Tuple[Tuple[int, int], int]: A tuple where the first element are the sizes of the lookback period and the size of the input state,
                                         and the second element is the size of the label (next state).

        Raises:
            ValueError: If the dataset has no samples.
        """
        sample = self[0]
        if sample is None:
            raise ValueError("Dataset has no samples.")
        return ((self.lookback, sample[0].size(1)), sample[1].size(0))


class SequentialEventDataset(Dataset):
    # TODO: This class should return a new state based on the next event in the sequence
    #       If the event happened after the lookahead time (5min), then use the current state as label (no change)
    #       If the event happened before or at the lookahead time (5min), then use the new changed state as label (state changed)
    def __init__(
        self, 
        csv_file: str, 
        n_columns: int, 
        max_samples: Optional[int] = None, 
        lookback: int = 1, 
        lookahaed: int = 1, 
        device: str = "cpu"
    ) -> None:
        if lookback < 1:
            raise ValueError(f"Lookback <{lookback}> must be greater than 0")

        data: pd.DataFrame = pd.read_csv(
            csv_file,
            delimiter=",",
            header=0,
            usecols=range(n_columns),
            dtype="float32",
            nrows=max_samples,
            index_col=False,
        )
        self.columns: list[str] = data.columns.tolist()

        self.data: np.ndarray = data.to_numpy()

        self.transforms: Optional[Sequence[DataTransform]] = None
        self.lookback: int = lookback
        self.lookahead: int = lookahaed
        self.device = device

    def _calculate_timespan(self, first: np.ndarray, second: np.ndarray) -> float:
        minute_idx = self.columns.index("Minute")
        hour_idx = self.columns.index("Hour")
        month_idx = self.columns.index("Month")
        year_idx = self.columns.index("Year")

        return (
            (float(first[year_idx]) - float(second[year_idx])) * 365 * 31 * 24 * 60
            + (float(first[month_idx]) - float(second[month_idx])) * 31 * 24 * 60
            + (float(first[hour_idx]) - float(second[hour_idx])) * 60
            + (float(first[minute_idx]) - float(second[minute_idx]))
        )

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, index) -> None | tuple[torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError("Index out of bounds")

        next_event = self.data[index + self.lookback]
        sequence = self.data[index : index + self.lookback]
        next_event_timespan = self._calculate_timespan(sequence[-1], next_event)

        # Check if next state is changed by an event
        next_state = next_event
        if next_event_timespan > self.lookahead:
            next_state = sequence[-1]

        sample = (sequence, next_state)

        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)
                if sample is None:
                    return None

        sample = (
            torch.tensor(sample[0], dtype=torch.float32, device=self.device),
            torch.tensor(sample[1], dtype=torch.float32, device=self.device),
        )

        return sample

    # def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    #     for i in range(len(self)):
    #         sample = self[i]
    #         if sample is not None:
    #             yield sample

    def set_transforms(self, transforms: Optional[Sequence[DataTransform]] = None) -> None:
        self.transforms = transforms

    def sample_sizes(self) -> tuple[tuple[int, int], int]:
        """
        Retrieves the sample sizes from the dataset.

        Returns:
            Tuple[Tuple[int, int], int]: A tuple where the first element are the sizes of the lookback period and the size of the input state,
                                         and the second element is the size of the label (next state).

        Raises:
            ValueError: If the dataset has no samples.
        """
        sample = self[0]
        if sample is None:
            raise ValueError("Dataset has no samples.")
        return ((self.lookback, sample[0].size(1)), sample[1].size(0))

class RemoveColumnsTransform(DataTransform):
    """Removes the specified columns from samples.
    if remove_label_cols is None, it removes the same columns from both features and labels."""

    def __init__(self, remove_feature_cols: Sequence[int], remove_label_cols: Optional[Sequence[int]] = None):
        self.remove_feature_cols = remove_feature_cols
        self.remove_label_cols = remove_label_cols or remove_feature_cols

    @override
    def __call__(self, sample: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        sample = (np.delete(sample[0], self.remove_feature_cols, axis=1), np.delete(sample[1], self.remove_label_cols))
        return sample
