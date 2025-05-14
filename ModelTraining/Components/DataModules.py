"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: A PyTorch Lightning data module for handling sequential state datasets.
            It manages data loading and preprocessing for training, validation, and testing.
Date: 2025-05-14
"""


from typing import List

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from .Data.SequentialData import (DataTransform, RemoveColumnsTransform,
                                  SequentialStateDataset)


class SequenceDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        sequence_len: int,
        train_ds_path: str,
        val_ds_path: str,
        test_ds_path: str,
        n_data_columns: int,
        n_devices: int,
        n_users: int,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.train_ds_path = train_ds_path
        self.val_ds_path = val_ds_path
        self.test_ds_path = test_ds_path
        self.n_data_columns = n_data_columns
        self.n_devices = n_devices
        self.n_users = n_users
        self.num_workers = num_workers

        self.train_ds: SequentialStateDataset | None = None
        self.val_ds: SequentialStateDataset | None = None
        self.test_ds: SequentialStateDataset | None = None

        self.n_features: int | None = None

    def _get_transforms(self, ds: SequentialStateDataset) -> List[DataTransform]:
        time_columns = ["Minute", "Hour", "DayOfWeek", "DayOfMonth", "Month", "Year"]
        remove_feature_columns = [
            ds.columns.index("DayOfMonth"),
            ds.columns.index("Month"),
            ds.columns.index("Year"),
        ]
        return [
            RemoveColumnsTransform(
                remove_feature_cols=remove_feature_columns,
                remove_label_cols=range(len(time_columns) + self.n_users), # Remove time and user locations from labels
            )
        ]

    def prepare_data(self) -> None:
        self.train_ds = SequentialStateDataset(self.train_ds_path, self.n_data_columns, lookback=self.sequence_len)
        self.val_ds   = SequentialStateDataset(self.val_ds_path, self.n_data_columns, lookback=self.sequence_len)
        self.test_ds  = SequentialStateDataset(self.test_ds_path, self.n_data_columns, lookback=self.sequence_len)

        self.train_ds.set_transforms(self._get_transforms(self.train_ds))
        self.val_ds.set_transforms(self._get_transforms(self.val_ds))
        self.test_ds.set_transforms(self._get_transforms(self.test_ds))

        self.n_features = self.train_ds.sample_sizes()[0][1]

    def get_feature_size(self) -> int:
        if self.n_features is None:
            ds = SequentialStateDataset(self.train_ds_path, self.n_data_columns, lookback=self.sequence_len)
            ds.set_transforms(self._get_transforms(ds))
            self.n_features = ds.sample_sizes()[0][1]
            del ds

        return self.n_features

    def train_dataloader(self):
        if self.train_ds is None:
            return None
        return DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers = True,
            shuffle = False
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers = True,
            shuffle = False
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers = True,
            shuffle = False
        )