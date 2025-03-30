from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import (AUROC, Accuracy, F1Score, HammingDistance,
                          MetricCollection, Precision, Recall)

from .Metrics.StateChangeAccuracy import StateChangeAccuracy
from .ModelBase import ModelBase


class SequenceLearner(pl.LightningModule):
    def __init__(self, model: ModelBase | nn.Module, lr: float, keep_hidden_state: bool, n_devices: int, sequence_len: int):
        super().__init__()
        self.model = model
        self.lr = lr
        self.keep_hidden_state = keep_hidden_state
        self.n_devices = n_devices
        self.sequence_len = sequence_len
        self.loss_fn = nn.MSELoss()
        self.previous_hidden: torch.Tensor | None = None

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.validation_epoch_start_step = 0
        
        # Add a step counter for periodic logging
        self.log_every_n_steps = 10

        self.added_custom_scalers: bool = False

        self.accuracy = Accuracy(task="multilabel", num_classes=2, num_labels=n_devices)
        self.precision = Precision(task="multilabel", num_classes=2, num_labels=n_devices)
        self.recall = Recall(task="multilabel", num_classes=2, num_labels=n_devices)
        self.f1score = F1Score(task="multilabel", num_classes=2, num_labels=n_devices)
        self.auroc = AUROC(task="multilabel", num_classes=2, num_labels=n_devices)
        self.hamming = HammingDistance(task="multilabel", num_classes=2, num_labels=n_devices)

        self.device_accuracies = nn.ModuleList([Accuracy(task="binary") for _ in range(self.n_devices)])
        self.device_state_change_accuracies = nn.ModuleList([StateChangeAccuracy() for _ in range(self.n_devices)])
        self.device_precisions = nn.ModuleList([Precision(task="binary") for _ in range(self.n_devices)])
        self.device_recalls = nn.ModuleList([Recall(task="binary") for _ in range(self.n_devices)])

        if isinstance(self.model, ModelBase):
            self.hparams.update(self.model.get_hparams())
        self.save_hyperparameters(ignore=["model"])

    def add_custom_scalers(self):
        if self.added_custom_scalers:
            return
        self.added_custom_scalers = True

        for writer in [logger.experiment for logger in self.loggers if isinstance(logger.experiment, SummaryWriter)]: # type: ignore
            writer.add_custom_scalars({
                "Pred_vs_Actual": {
                    f"Device {d}": [
                        "Multiline", [
                            f"val_actual_device_{d}",
                            f"val_pred_device_{d}"
                        ]
                    ] for d in range(self.n_devices)},
                "Metrics": {
                    "Accuracy": [
                        "Multiline", [
                            "train_accuracy",
                            "val_accuracy"
                        ]
                    ],
                    "Precision": [
                        "Multiline", [
                            "train_precision",
                            "val_precision"
                        ]
                    ],
                    "Recall": [
                        "Multiline", [
                            "train_recall",
                            "val_recall"
                        ]
                    ],
                    "F1score": [
                        "Multiline", [
                            "train_f1score",
                            "val_f1score"
                        ]
                    ],
                    "Auroc": [
                        "Multiline", [
                            "train_auroc",
                            "val_auroc"
                        ]
                    ],
                    "Hamming": [
                        "Multiline", [
                            "train_hamming",
                            "val_hamming"
                        ]
                    ],
                }
            })

    # Step

    def training_step(self, batch, batch_idx):
        out = self._base_step(batch, batch_idx)
        # self.training_step_outputs.append(out)

        self.log_dict({"train_loss": out["loss"]}, on_step=True, on_epoch=False, prog_bar=False)
        
        y_hat = out["y_hat"]
        y = out["y"]
        
        y_hat_clamp = y_hat.clamp(0.0, 1.0)
        y_hat_rounded = y_hat_clamp >= 0.5
        y_int = y.int()
        
        self.log_dict({
            "train_accuracy": self.accuracy(y_hat_rounded, y_int),
            "train_precision": self.precision(y_hat_rounded, y_int),
            "train_recall": self.recall(y_hat_rounded, y_int),
            "train_f1score": self.f1score(y_hat_rounded, y_int),
            "train_auroc": self.auroc(y_hat_clamp, y_int),
            "train_hamming": self.hamming(y_hat_clamp, y_int)
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        for d in range(self.n_devices):
            state_change_metric = self.device_state_change_accuracies[d](y_hat_rounded[:, d], y_int[:, d])
            
            self.log_dict({
                f"train_device_{d}_accuracy": self.device_accuracies[d](y_hat_rounded[:, d], y_int[:, d]),
                f"train_device_{d}_state_change_accuracy": state_change_metric["accuracy"],
                f"train_device_{d}_total_state_changes": state_change_metric["total_state_changes"],
                f"train_device_{d}_precision": self.device_precisions[d](y_hat_rounded[:, d], y_int[:, d]),
                f"train_device_{d}_recall": self.device_recalls[d](y_hat_rounded[:, d], y_int[:, d])
            }, on_step=True, on_epoch=True, prog_bar=False)
        
        return out

    def validation_step(self, batch, batch_idx):
        out = self._base_step(batch, batch_idx)
        self.validation_step_outputs.append(out)

        return out

    def test_step(self, batch, batch_idx):
        # Just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def _base_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, y = batch

        y_hat: torch.Tensor
        c: torch.Tensor
        y_hat, c = self.model.forward(x, self.previous_hidden)

        y_hat = y_hat[:, -1, :]   # Take the last prediction in the sequence
        loss = self.loss_fn(y_hat, y)

        if self.keep_hidden_state:
            self.previous_hidden = c.detach()

        return {"loss": loss, "y": y, "y_hat": y_hat}

    # Epoch end

    def on_train_epoch_end(self) -> None:
        self._on_base_epoch_end()

        # y_hat = torch.cat([out["y_hat"] for out in self.training_step_outputs])
        # y = torch.cat([out["y"] for out in self.training_step_outputs])

        # y_hat_clamp = y_hat.clamp(0.0, 1.0)
        # y_hat_rounded = y_hat_clamp >= 0.5
        # y_int = y.int()

        # self.log_dict({
        #     "train_accuracy": self.accuracy(y_hat_rounded, y_int),
        #     "train_precision": self.precision(y_hat_rounded, y_int),
        #     "train_recall": self.recall(y_hat_rounded, y_int),
        #     "train_f1score": self.f1score(y_hat_rounded, y_int),
        #     "train_auroc": self.auroc(y_hat_clamp, y_int),
        #     "train_hamming": self.hamming(y_hat_clamp, y_int)
        # }, on_step=False, on_epoch=True, prog_bar=True)

        # for d in range(self.n_devices):
        #     state_change_metric = self.device_state_change_accuracies[d](y_hat_rounded[:, d], y_int[:, d])

        #     self.log_dict({
        #         f"train_device_{d}_accuracy": self.device_accuracies[d](y_hat_rounded[:, d], y_int[:, d]),
        #         f"train_device_{d}_state_change_accuracy": state_change_metric["accuracy"], # type: ignore
        #         f"train_device_{d}_total_state_changes": state_change_metric["total_state_changes"], # type: ignore
        #         f"train_device_{d}_precision": self.device_precisions[d](y_hat_rounded[:, d], y_int[:, d]),
        #         f"train_device_{d}_recall": self.device_recalls[d](y_hat_rounded[:, d], y_int[:, d])
        #     }, on_step=False, on_epoch=True, prog_bar=False)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self._on_base_epoch_end()

        y_hat = torch.cat([out["y_hat"] for out in self.validation_step_outputs])
        y = torch.cat([out["y"] for out in self.validation_step_outputs])

        if not self.trainer.sanity_checking:
            for s in range(y.size(0)): # steps
                for d in range(self.n_devices): # devices
                    for logger in self.loggers:
                        logger.log_metrics({
                            f"val_actual_device_{d}": y[s, d].item(),
                            f"val_pred_device_{d}": y_hat[s, d].item()
                        }, step=self.validation_epoch_start_step + s)
            self.validation_epoch_start_step += y.size(0)

        y_hat_clamp = y_hat.clamp(0.0, 1.0)
        y_hat_rounded = y_hat_clamp >= 0.5
        y_int = y.int()

        self.log_dict({
            "val_accuracy": self.accuracy(y_hat_rounded, y_int),
            "val_precision": self.precision(y_hat_rounded, y_int),
            "val_recall": self.recall(y_hat_rounded, y_int),
            "val_f1score": self.f1score(y_hat_rounded, y_int),
            "val_auroc": self.auroc(y_hat_clamp, y_int),
            "val_hamming": self.hamming(y_hat_clamp, y_int)
        }, on_step=False, on_epoch=True, prog_bar=True)

        for d in range(self.n_devices):
            state_change_metric = self.device_state_change_accuracies[d](y_hat_rounded[:, d], y_int[:, d])

            self.log_dict({
                f"val_device_{d}_accuracy": self.device_accuracies[d](y_hat_rounded[:, d], y_int[:, d]),
                f"val_device_{d}_state_change_accuracy": state_change_metric["accuracy"],
                f"val_device_{d}_total_state_changes": state_change_metric["total_state_changes"],
                f"val_device_{d}_precision": self.device_precisions[d](y_hat_rounded[:, d], y_int[:, d]),
                f"val_device_{d}_recall": self.device_recalls[d](y_hat_rounded[:, d], y_int[:, d])
            }, on_step=False, on_epoch=True, prog_bar=False)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._on_base_epoch_end()

        # Just reuse the on_validation_epoch_end
        self.on_validation_epoch_end()

    def on_predict_epoch_end(self) -> None:
        self._on_base_epoch_end()

    def _on_base_epoch_end(self):
        self.previous_hidden = None

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1score.reset()
        self.auroc.reset()
        self.hamming.reset()

        for d in range(self.n_devices):
            self.device_accuracies[d].reset()
            self.device_state_change_accuracies[d].reset()
            self.device_precisions[d].reset()
            self.device_recalls[d].reset()

    # Epoch start

    def on_train_epoch_start(self) -> None:
        self._on_base_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self._on_base_epoch_start()

    def on_test_epoch_start(self) -> None:
        self._on_base_epoch_start()

    def on_predict_epoch_start(self) -> None:
        self._on_base_epoch_start()

    def _on_base_epoch_start(self):
        self.previous_hidden = None

    # Optimizers

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
