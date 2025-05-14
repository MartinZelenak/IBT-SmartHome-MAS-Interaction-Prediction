"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: A custom PyTorch metric that measures prediction accuracy specifically at points
            where device states change. It tracks state transitions across sequential data points
            and evaluates the accuracy at these moments.
Date: 2025-05-14
"""


import torch
from torchmetrics import Metric


class StateChangeAccuracy(Metric):
    """
    A Metric that measures accuracy specifically at points where the state changes.

    This metric tracks changes in the target values across sequential data points and measures
    how accurately a model predicts these state transitions.

    Attributes:
        last_batch_target (torch.Tensor): Stores the last target value(s) from the previous call.
        total_state_changes (torch.Tensor): Counter for total number of state changes detected.
        correct_state_changes (torch.Tensor): Counter for correctly predicted state changes.

    Example:
        >>> metric = StateChangeAccuracy()
        >>> pred = torch.tensor([[0, 1], [1, 1], [1, 0]])
        >>> target = torch.tensor([[0, 1], [0, 1], [1, 0]])
        >>> for i in range(len(pred)):
        >>>     metric.update(pred[i:i+1], target[i:i+1])
        >>> metric.compute()
        {'state_change_accuracy': tensor(0.5000), 'total_state_changes': tensor(2)}
    """

    def __init__(self):
        super().__init__()
        self.add_state("last_batch_target", default=torch.empty(0), persistent=True)
        self.add_state("total_state_changes", default=torch.tensor(0.0), persistent=True)
        self.add_state("correct_state_changes", default=torch.tensor(0.0), persistent=True)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric states with new predictions and targets.

        This method detects state changes by comparing current targets with previous ones,
        both within and across calls. It counts state changes and checks if predictions
        were correct at those change points.

        Args:
            pred (torch.Tensor): Model predictions
            target (torch.Tensor): Ground truth targets with same shape as pred.
        """
        # Flatten the inputs
        pred = pred.view(-1)
        target = target.view(-1)

        if self.last_batch_target.numel() > 0:
            extended_target = torch.cat([self.last_batch_target.unsqueeze(0), target])
            prev_targets = extended_target[:-1]
            cur_targets = target
            cur_preds = pred
        else:
            # Skip the first element as it has no previous value to compare with
            prev_targets = target[:-1]
            cur_targets = target[1:]
            cur_preds = pred[1:]

        # State change positions
        state_changes = ~torch.eq(cur_targets, prev_targets)

        self.total_state_changes += torch.sum(state_changes).float()

        # Correct predictions at state changes
        correct_at_changes = torch.eq(cur_preds, cur_targets) & state_changes
        self.correct_state_changes += torch.sum(correct_at_changes).float()

        if target.size(0) > 0:
            self.last_batch_target = target[-1].detach().clone()

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute the final metric value from the accumulated states.

        Returns:
            dict: A dictionary containing:
                - 'accuracy': The proportion of correctly predicted state changes
                - 'total_state_changes': The total number of state changes observed
        """
        return {
            'accuracy': self.correct_state_changes.float() / max(1, self.total_state_changes),  # type: ignore
            'total_state_changes': self.total_state_changes
        }
