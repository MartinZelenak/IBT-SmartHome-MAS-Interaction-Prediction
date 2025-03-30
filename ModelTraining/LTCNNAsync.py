import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

import SequentialData
import visualisation.plot as plot


class PerDeviceLTCAsync(nn.Module):

    def __init__(self, shared_input_size: int, per_device_input_size: int, n_devices: int, n_neurons: int, lookahead: int):
        """
        Initializes the PerDeviceLTCAsync model with specified parameters for shared and device-specific inputs.

        This constructor sets up the model's architecture by defining the sizes of shared and per-device inputs, 
        the number of devices, the number of neurons, and the lookahead parameter.  
        It also creates a list of device-specific models, each configured with the appropriate input sizes and neuron settings.

        Args:
            shared_input_size (int): The size of the shared input across devices.
            per_device_input_size (int): The size of the input specific to each device.
            n_devices (int): The number of devices to be used in the model.
            n_neurons (int): The number of neurons in the device models.
            lookahead (int): The lookahead parameter for the model's predictions.
        """
        super(PerDeviceLTCAsync, self).__init__()

        self.shared_input_size = shared_input_size
        self.per_device_input_size = per_device_input_size
        self.n_devices = n_devices
        self.n_neurons = n_neurons
        self.lookahead_t = torch.tensor([lookahead])

        self.device_models = nn.ModuleList(
            [
                LTC(shared_input_size + per_device_input_size, AutoNCP(n_neurons, 1), batch_first=True)
                for _ in range(n_devices)
            ]
        )

    def forward(self, input: torch.Tensor, timespans: torch.Tensor, hidden_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forwards the given input through a LTC model for each device

        Args:
            input (torch.Tensor): Input with dimensions [batch_size, sequence_len, shared_input_size + (per_device_input_size * n_devices)]
            timespans (torch.Tensor): The times between states and the lookahead time as the last number with dimensions [batch_size, sequence_len]
            hidden_state (torch.Tensor | None): Hidden state with dimensions [n_devices, batch_size, n_neurons] or None

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output with dimensions [batch_size, sequence_len, n_devices] and new hidden state with dimensions [n_devices, batch_size, n_neurons]
        """
        batch_size, sequence_len, _ = input.size()
        output = torch.zeros((batch_size, sequence_len, self.n_devices), device=input.device)

        if hidden_state is None:
            hidden_state = torch.zeros((self.n_devices, batch_size, self.n_neurons), device=input.device)

        out_hidden_state = torch.zeros_like(hidden_state)

        for i in range(self.n_devices):
            device_input = torch.cat(
                [
                    input[:, :, : self.shared_input_size],
                    input[:, :, self.shared_input_size + i : self.shared_input_size + i + self.per_device_input_size],
                ],
                dim=2,
            )
            device_output, out_hidden_state[i] = self.device_models[i].forward(device_input, hidden_state[i], timespans)
            output[:, :, i] = device_output.squeeze(-1)

        return output, out_hidden_state


def calculate_timespans(sequence: torch.Tensor, time_column_list: List[str], lookahead: float | None = None) -> torch.Tensor:
    """Calculates the time between states for given sequence.

    Args:
        sequence (torch.Tensor): State sequence with shape (sequence_len, state_size)
        time_column_list (List[str]): List of time column names.
                                      Expecting to find values: "Minute", "Hour", "Month", "Year".
        lookahead (float | None): Appended as the last timespan

    Returns:
        torch.Tensor: Timespans between states with shape (sequence_len - 1)
    """

    minute_idx = time_column_list.index("Minute")
    hour_idx = time_column_list.index("Hour")
    month_idx = time_column_list.index("Month")
    year_idx = time_column_list.index("Year")

    timespans = torch.zeros([sequence.size(0)] if lookahead else [sequence.size(0) - 1])
    prev_time: Tuple[float, float, float, float] | None = None
    for i, state in enumerate(sequence):
        cur_time = (
            state[minute_idx].item(),
            state[hour_idx].item(),
            state[month_idx].item(),
            state[year_idx].item(),
        )
        if prev_time is not None:
            delta_minutes = (
                (cur_time[3] - prev_time[3]) * 365 * 31 * 24 * 60 +
                (cur_time[2] - prev_time[2]) * 31 * 24 * 60 +
                (cur_time[1] - prev_time[1]) * 60 +
                (cur_time[0] - prev_time[0])
            )
            timespans[i - 1] = delta_minutes

        prev_time = cur_time
    
    if lookahead:
        timespans[-1] = lookahead

    return timespans


if __name__ == "__main__":
    device = "cpu"
    print(f"Device: {device}")

    # Create out folder
    ## Use script name without .py extension
    out_folder = f"out/{os.path.basename(__file__)[:-3]}"
    os.makedirs(out_folder, exist_ok=True)
    print(f"Output folder: {out_folder}")

    # Dataset
    ## Parameters
    train_data_file_path = "./datasets/year-events.csv"
    eval_data_file_path = "./datasets/month-events.csv"
    time_column_names = ["Minute", "Hour", "DayOfWeek", "DayOfMonth", "Month", "Year"]
    n_total_time_columns = len(time_column_names)
    n_SmartDevices = 8
    remove_feature_columns = [
        "DayOfMonth",
        "Month",
        "Year",
    ]  # Remove unused feature columns

    max_samples = None
    sequence_len = 6  # Event sequence length
    lookahead = 5 # minutes

    ## Load data
    n_data_columns = (
        n_total_time_columns + 1 + n_SmartDevices
    )  # timeColumns + location + deviceStates
    n_time_columns = n_total_time_columns - len(
        [col for col in remove_feature_columns if col in time_column_names]
    )

    train_dataset = SequentialData.SequentialEventDataset(
        train_data_file_path, n_data_columns, max_samples, lookback=sequence_len, lookahaed=lookahead, device=device
    )
    eval_dataset = SequentialData.SequentialEventDataset(
        eval_data_file_path, n_data_columns, max_samples, lookback=sequence_len, lookahaed=lookahead, device=device
    )

    ## Data transforms
    transforms = [
        ### Remove unused columns
        SequentialData.RemoveColumnsTransform(
            remove_feature_cols=[
                train_dataset.columns.index(col_name) for col_name in remove_feature_columns
            ],
            remove_label_cols=range(n_total_time_columns + 1),
        )
    ]

    train_dataset.set_transforms(transforms)
    eval_dataset.set_transforms(transforms)
    n_feature_columns: int = train_dataset.sample_sizes()[0][1]

    # Hyper-parameters
    ## Model
    n_neurons = 16
    output_size = 1
    shared_input_size = n_feature_columns - n_SmartDevices
    ## Training
    load_model = False
    n_epochs = 1
    learning_rate = 0.001

    # Model
    ## Model: Sequence of previous states -> next device states
    ## State: [timeColumns, location, deviceStates]
    model = PerDeviceLTCAsync(
        shared_input_size=shared_input_size,  # features without device states
        per_device_input_size=1,              # device state (on/off)
        n_devices=n_SmartDevices,
        n_neurons=n_neurons,
        lookahead=lookahead
    ).to(device)

    # Load or train model
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    if load_model:
        model.load_state_dict(torch.load(f"{out_folder}/{LTC.__name__}.pth"))
    else:
        # Loss and optimizer
        criterion_class = nn.MSELoss
        criterion = criterion_class(reduction="sum")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        start_time_total = time.time()
        start_time = time.time()
        losses: List[Tuple[float, bool]] = []   # List of losses with is_state_change flag
        correct_device_state_changes_per_week: List[int] = [0 for _ in range(n_SmartDevices)] # TODO: Calculate and visualize
        model.train()
        for epoch in range(n_epochs):
            for i, data in enumerate(train_dataset):    # type: ignore
                sequence: torch.Tensor
                next_state: torch.Tensor
                sequence, next_state = data
                current_state = sequence[-1]
                timespans = calculate_timespans(sequence, time_column_names, lookahead).to(device)

                pred, _ = model.forward(sequence.unsqueeze(0), timespans.unsqueeze(0))
                loss: torch.Tensor = criterion(pred[:, [-1], :], next_state[None, None])
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record loss
                if not torch.equal(current_state[n_time_columns + 1 :], next_state):
                    losses.append((loss.item(), True))
                else:
                    losses.append((loss.item(), False))

                # Print progress
                if (i + 1) % 100 == 0:
                    print(f"\rEpoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(train_dataset)}], Loss: {loss.item():.4f}", end="")
            print(f"\rEpoch [{epoch + 1}/{n_epochs}], Step [{len(train_dataset)}/{len(train_dataset)}], Loss: {loss.item():.4f}")

        ## Saving losses
        with open(f"{out_folder}/losses.csv", "w") as f:
            f.write("Loss,IsStateChange\n")
            f.writelines([f"{loss[0]},{1 if loss[1] else 0}\n" for loss in losses])

        ## Plot loss
        print("Plotting loss...")
        plot.plot_losses([loss[0] for loss in losses], f"{out_folder}/loss.png", "Loss")
        plot.plot_losses([loss[0] for loss in losses if not loss[1]], f"{out_folder}/sameStateLoss.png", "Same State Loss")
        plot.plot_losses([loss[0] for loss in losses if loss[1]], f"{out_folder}/stateChangeLoss.png", "State Change Loss")
        plot_average_window_size = 100
        plot.plot_average_loss(
            [loss[0] for loss in losses],
            plot_average_window_size,
            f"{out_folder}/averageLoss.png",
            f"Average loss per {plot_average_window_size} events",
            "Window",
            "Average loss",
        )

        # Save model
        print('Saving model...')
        torch.save(model.state_dict(), f"{out_folder}/{LTC.__name__}.pth")
        ## Save parameters
        with open(f"{out_folder}/parameters.txt", "w") as f:
            f.write(f"Model Hyperparameters:\n")
            f.write(f"n_neurons: {n_neurons}\n")
            f.write(f"output_size: {output_size}\n")
            f.write(f"shared_input_size: {shared_input_size}\n")
            f.write(f"per_device_input_size: 1\n")
            f.write(f"n_devices: {n_SmartDevices}\n")
            f.write(f"\nTraining Parameters:\n")
            f.write(f"sequence_len: {sequence_len}\n")
            f.write(f"n_epochs: {n_epochs}\n")
            f.write(f"learning_rate: {learning_rate}\n")
            f.write(f"loss function: {criterion_class.__name__}\n")
        print('Model saved')

    # Evaluate
    print('Evaluating model...')
    devices_pred_actual: List[List[Tuple[float, float]]] = [[] for _ in range(n_SmartDevices)] # [[(pred, actual), ...], ...]
    correct_guesses: int = 0
    total_state_changes: int = 0
    correct_state_change_guesses: int = 0
    devices_state_changes: List[int] = [0 for _ in range(n_SmartDevices)]
    correct_device_state_changes: List[int] = [0 for _ in range(n_SmartDevices)]
    correct_device_state_changes_with_lag: List[int] = [0 for _ in range(n_SmartDevices)]
    with torch.no_grad():
        for i, data in enumerate(eval_dataset): # type: ignore
            sequence: torch.Tensor
            next_state: torch.Tensor
            sequence, next_state = data
            timespans = calculate_timespans(sequence, time_column_names, lookahead).to(device)
            current_state = sequence[-1]
            previous_state = sequence[-2] if len(sequence) > 1 else current_state

            # Predict
            pred, _ = model.forward(sequence.unsqueeze(0), timespans.unsqueeze(0))
            pred = pred[0, -1, :] # take the last prediction in the sequence [batch, sequence, device_pred]

            # Save predicted and actual state
            for j in range(n_SmartDevices):
                devices_pred_actual[j].append((float(pred[j]), float(next_state[j])))
            del j

            # Compare prediction with actual state
            ## Whole system state
            pred = torch.round(pred)
            pred[pred == -0.0] = 0.0
            pred_correct: bool = torch.equal(pred, next_state)
            state_changed: bool = not torch.equal(current_state[-n_SmartDevices:], next_state)
            if pred_correct:
                correct_guesses += 1
            if state_changed:
                total_state_changes += 1
                if pred_correct:
                    correct_state_change_guesses += 1

            ## Per device state
            for j in range(n_SmartDevices):
                # Check if device state changed
                ## correct_device_state_changes
                if not torch.equal(current_state[-n_SmartDevices + j], next_state[j]):
                    devices_state_changes[j] += 1
                    if torch.equal(pred[j], next_state[j]): # TODO: Specify prediction threshold?
                        correct_device_state_changes[j] += 1
                # Check if previous device state changed and prediction was late
                ## correct_device_state_changes_with_lag
                if (
                    previous_state is not None
                    and not torch.equal(previous_state[-n_SmartDevices + j], current_state[-n_SmartDevices + j]) # State previously changed for this device
                    and torch.equal(pred[j], current_state[-n_SmartDevices + j])                                 # The change corresponds with the prediction (with lag)
                ):
                    correct_device_state_changes_with_lag[j] += 1
            del j

            # Print progress
            if (i + 1) % 100 == 0:
                print(f'\rStep [{i + 1}/{len(eval_dataset)}]', end="")
        print(f'\rStep [{len(eval_dataset)}/{len(eval_dataset)}]')

    # Print accuracy
    with open(f'{out_folder}/accuracy.txt', 'w') as f:
        to_print: str = f'Correct guesses: {correct_guesses}/{len(eval_dataset)} ({(correct_guesses/len(eval_dataset))*100:.2f}%)\n'
        to_print += f'Correct state change guesses: {correct_state_change_guesses}/{total_state_changes} ({(correct_state_change_guesses/total_state_changes)*100:.2f}%)\n'
        print(to_print, end="")
        f.write(to_print)
        for i in range(n_SmartDevices):
            if devices_state_changes[i] == 0:
                to_print = f'Device {i} had no state changes\n'
                print(to_print, end="")
                f.write(to_print)
                continue
            to_print = f'Device {i} correct state change guesses: {correct_device_state_changes[i]}/{devices_state_changes[i]} ({((correct_device_state_changes[i]/devices_state_changes[i]) * 100):.2f}%)\n'
            print(to_print, end="")
            f.write(to_print)
    del i

    # Plot predicted vs actual
    n_zoomed_samples: int = len(devices_pred_actual[0]) // 4
    n_zoomed_state_change_samples: int = 100

    print('Plotting predicted vs actual...')
    for i in range(n_SmartDevices):
        # Whole evaluation
        plt.figure(figsize=(20, 6))
        plt.suptitle(f'Device {i}: Predicted vs Actual')

        plt.subplot(2,1,1)
        plt.xlabel('Time Step')
        plt.ylabel('Actual')
        plt.scatter(range(len(devices_pred_actual[i])), [x[1] for x in devices_pred_actual[i]], s=3)

        plt.subplot(2,1,2)
        plt.xlabel('Time Step')
        plt.ylabel('Predicted')
        plt.plot(range(len(devices_pred_actual[i])), [x[0] for x in devices_pred_actual[i]])

        plt.savefig(f'{out_folder}/device{i}PredActual.png')
        plt.close()

        # Zoomed
        plt.figure(figsize=(20, 6))
        plt.suptitle(f'Device {i}: Predicted vs Actual (First 1/4)')

        plt.subplot(2,1,1)
        plt.xlabel('Time Step')
        plt.ylabel('Actual')
        plt.scatter(range(n_zoomed_samples), [x[1] for x in devices_pred_actual[i][:n_zoomed_samples]], s=3)

        plt.subplot(2,1,2)
        plt.xlabel('Time Step')
        plt.ylabel('Predicted')
        plt.plot(range(n_zoomed_samples), [x[0] for x in devices_pred_actual[i][:n_zoomed_samples]])

        plt.savefig(f'{out_folder}/device{i}PredActual-Zoom.png')
        plt.close()

        # Zoomed on state change
        ## Find first state change
        state_change_index: int|None = None
        prev_sample = devices_pred_actual[i][0]
        for j, sample in enumerate(devices_pred_actual[i]):
            if sample[1] != prev_sample[1]:
                state_change_index = j
                break
            prev_sample = sample
        del j

        if state_change_index is not None:
            start_index = state_change_index - (n_zoomed_state_change_samples // 2)
            start_index = max(start_index, 0)
            end_index = state_change_index + (n_zoomed_state_change_samples // 2)
            end_index = min(end_index, len(devices_pred_actual[i]))

            plt.figure(figsize=(20, 6))
            plt.suptitle(f'Device {i}: Predicted vs Actual (State change zoom)')

            plt.subplot(2,1,1)
            plt.axvline(
                x=state_change_index - start_index,
                color="red",
                linestyle="--",
                label="State change",
            )
            plt.xlabel('Time Step')
            plt.ylabel('Actual')
            plt.scatter(range((end_index - start_index) + 1), [x[1] for x in devices_pred_actual[i][start_index : (end_index + 1)]], s=3)

            plt.subplot(2,1,2)
            plt.axvline(
                x=state_change_index - start_index,
                color="red",
                linestyle="--",
                label="State change",
            )
            plt.xlabel('Time Step')
            plt.ylabel('Predicted')
            plt.plot(range((end_index - start_index) + 1), [x[0] for x in devices_pred_actual[i][start_index : (end_index + 1)]])

            plt.savefig(f'{out_folder}/device{i}PredActual-State-Change-Zoom.png')
            plt.close()

    print('Done')
