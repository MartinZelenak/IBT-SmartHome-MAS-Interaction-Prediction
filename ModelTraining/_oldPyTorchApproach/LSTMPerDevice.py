import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import Compose as TransformsCompose

import Data


class PerDeviceLSTM(nn.Module):
    def __init__(self, shared_input_size, per_device_input_size, output_size, hidden_size, num_layers):
        super(PerDeviceLSTM, self).__init__()
        
        self.shared_input_size = shared_input_size
        self.per_device_input_size = per_device_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.deviceModels = nn.ModuleList([
            nn.ModuleList([
                nn.LSTM(shared_input_size + per_device_input_size, hidden_size, num_layers, batch_first=True),
                nn.Linear(hidden_size, 1)
            ]) for _ in range(output_size)
        ])

    def forward(self, shared_input: torch.Tensor, per_device_inputs: List[torch.Tensor]) -> torch.Tensor:
        # TODO: Takes too long to train
        out = torch.Tensor(size=(shared_input.size(0), self.output_size)).to(shared_input.device)  # [sequence_len, output_size]
        for i in range(self.output_size):
            cur = torch.cat([shared_input, per_device_inputs[i].reshape([shared_input.size(0), self.per_device_input_size])], dim=1)
            cur, _ = self.deviceModels[i][0](cur)              # LSTM -> [sequence_len, hidden_size] # type: ignore
            out[:, i] = self.deviceModels[i][1](cur).squeeze() # Linear # type: ignore
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Create out folder
    outFolder = 'out/PerDeviceLSTM'
    os.makedirs(outFolder, exist_ok=True)
    print(f'Output folder: {outFolder}')


    # Hyper-parameters
    ## Dataset
    train_data_file_path = './datasets/year-5min.csv'
    eval_data_file_path = './datasets/month-5min.csv'
    time_column_names = ['Minute', 'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'Year']
    n_total_time_columns = len(time_column_names)
    n_SmartDevices = 8
    remove_feature_columns = ['DayOfMonth', 'Month', 'Year'] # Remove unused feature columns
    max_samples = None

    n_data_columns = n_total_time_columns + 1 + n_SmartDevices # timeColumns + location + deviceStates
    column_name_number_map = Data.get_column_name_number_map(train_data_file_path, 15)
    n_time_columns = n_total_time_columns - len([col for col in remove_feature_columns if col in time_column_names])

    transforms = []
    ### Remove unused columns
    transforms.append(Data.RemoveColumnsTransform(remove_feature_cols=[column_name_number_map[col_name] for col_name in remove_feature_columns],  # Remove feature columns
                                                  remove_label_cols=range(n_total_time_columns + 1)))                                          # Remove time columns and location
    # ### Convert time columns to slot number of day
    # transforms.append(Data.MinuteHourToSlotNumberOfDayTransform(minute_column=column_name_number_map['Minute'], 
    #                                           hour_column=column_name_number_map['Hour'], 
    #                                           minutes_between_slots=5, 
    #                                           only_current_state=True))  # Convert minute and hour to slot number of day
    # n_time_columns = n_time_columns - 1 # Remove SlotNumberOfDayReduction
    transform = TransformsCompose(transforms)

    ## Model
    hidden_size = 16
    num_layers = 2
    sequence_len = 6 # Time sequence length
    ## Training
    load_model = True
    n_epochs = 1
    learning_rate = 0.001

    # Dataset
    train_dataset = Data.NextStateDataset(train_data_file_path, max_samples=max_samples, n_state_cols=n_data_columns, transform=transform, device=device)
    eval_dataset = Data.NextStateDataset(eval_data_file_path, max_samples=max_samples, n_state_cols=n_data_columns, transform=transform, device=device)
    n_feature_columns: int = train_dataset.sample_sizes()[0]

    # Model
    ## Model: Sequence of previous states -> next device states
    ## State: [timeColumns, location, deviceStates]
    shared_input_size = n_feature_columns - n_SmartDevices
    model = PerDeviceLSTM(shared_input_size=shared_input_size,    # features without device states
                            per_device_input_size=1,              # device state (on/off)
                            hidden_size=hidden_size,
                            output_size=n_SmartDevices,
                            num_layers=num_layers).to(device)
    # Load trained model
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    if load_model:
        model.load_state_dict(torch.load(f'{outFolder}/{PerDeviceLSTM.__name__}.pth'))
    else:
        # Loss and optimizer
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        losses: List[float] = []
        stateChangeLosses: List[float] = []
        sameStateLosses: List[float] = []
        model.train()
        for epoch in range(n_epochs):
            sequence: torch.Tensor = torch.Tensor(size=(0,n_feature_columns)).to(device)
            for i, (currentState, nextState) in enumerate(train_dataset): # type: ignore
                sequence = torch.cat((sequence, currentState.unsqueeze(0)), 0)
                if sequence.size(0) >= sequence_len:
                    sequence = sequence[-sequence_len:] # keep only last sequence_len samples

                # Predict
                shared_input = sequence[:, :shared_input_size]
                per_device_inputs = [sequence[:, shared_input_size+i] for i in range(n_SmartDevices)]
                pred = model(shared_input, per_device_inputs)[-1] # last sample prediction (next state)
                loss = criterion(pred, nextState)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record loss
                losses.append(loss.item())
                if not torch.equal(currentState[n_time_columns+1:], nextState):
                    stateChangeLosses.append(loss.item())
                else:
                    sameStateLosses.append(loss.item())

                # Print progress
                if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_dataset)}], Loss: {loss.item():.4f}')

        # Plot loss
        print('Plotting loss...')
        plt.figure(figsize=(20, 6))
        plt.scatter(range(len(losses)), losses, s=3)
        plt.title('Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(f'{outFolder}/loss.png')

        plt.figure(figsize=(20, 6))
        plt.scatter(range(len(sameStateLosses)), sameStateLosses, s=3)
        plt.title('Same State Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(f'{outFolder}/sameStateLoss.png')

        plt.figure(figsize=(20, 6))
        plt.scatter(range(len(stateChangeLosses)), stateChangeLosses, s=3)
        plt.title('State Change Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(f'{outFolder}/stateChangeLoss.png')
        plt.close()

        # Save model
        print('Saving model...')
        torch.save(model.state_dict(), f'{outFolder}/{PerDeviceLSTM.__name__}.pth')
        print('Model saved')

    # Evaluate
    print('Evaluating model...')
    devicesPredActual: List[List[Tuple[float, float]]] = [[] for _ in range(n_SmartDevices)] # [[(pred, actual), ...], ...]
    correct_guesses: int = 0
    total_state_changes: int = 0
    correct_state_change_guesses: int = 0
    devices_state_changes: List[int] = [0 for _ in range(n_SmartDevices)]
    correct_device_state_changes: List[int] = [0 for _ in range(n_SmartDevices)]
    with torch.no_grad():
        sequence: torch.Tensor = torch.empty((0,n_feature_columns), device=device)
        for i, (currentState, nextState) in enumerate(eval_dataset): # type: ignore
            sequence = torch.cat((sequence, currentState.unsqueeze(0)), 0)
            if sequence.size(0) >= sequence_len:
                sequence = sequence[-sequence_len:] # keep only last sequence_len samples

            # Predict
            shared_input = sequence[:, :shared_input_size]
            per_device_inputs = [sequence[:, shared_input_size+i] for i in range(n_SmartDevices)]
            pred = model(shared_input, per_device_inputs)[-1] # last sample prediction (next state)

            # Save predicted and actual state
            for i in range(n_SmartDevices):
                devicesPredActual[i].append((float(pred[i]), float(nextState[i])))

            # Compare prediction with actual state
            pred = torch.round(pred)
            if torch.equal(pred, nextState):
                correct_guesses += 1
            if not torch.equal(currentState[-n_SmartDevices:], nextState):
                total_state_changes += 1
                if torch.equal(pred, nextState):
                    correct_state_change_guesses += 1
            for i in range(n_SmartDevices):
                if not torch.equal(currentState[-n_SmartDevices + i], nextState[i]):
                    devices_state_changes[i] += 1
                    if torch.equal(torch.round(pred[i]), nextState[i]): # TODO: Specify prediction threshold?
                        correct_device_state_changes[i] += 1

            # Print progress
            if (i+1) % 100 == 0:
                print(f'Step [{i+1}/{len(eval_dataset)}]')

    # Print accuracy
    with open(f'{outFolder}/accuracy.txt', 'w') as f:
        to_print: str = f'Correct guesses: {correct_guesses}/{len(eval_dataset)} ({(correct_guesses/len(eval_dataset))*100:.2f}%)\n'
        to_print += f'Correct state change guesses: {correct_state_change_guesses}/{total_state_changes} ({(correct_state_change_guesses/total_state_changes)*100:.2f}%)'
        print(to_print)
        f.write(to_print + '\n')
        for i in range(n_SmartDevices):
            if devices_state_changes[i] == 0:
                to_print = f'Device {i} had no state changes'
                print(to_print)
                f.write(to_print + '\n')
                continue
            to_print = f'Device {i} correct state change guesses: {correct_device_state_changes[i]}/{devices_state_changes[i]} ({(correct_device_state_changes[i]/devices_state_changes[i])*100:.2f}%)'
            print(to_print)
            f.write(to_print + '\n')

    # Plot predicted vs actual
    n_zoomed_samples = len(devicesPredActual[0]) // 4
    ## Find first state change
    n_zoomed_state_change_samples = 100
    state_change_index: int|None = None
    prevSample = devicesPredActual[0][0]
    for i, sample in enumerate(devicesPredActual[0]):
        if sample[1] != prevSample[1]:
            state_change_index = i
            break
        prevSample = sample
    del i

    print('Plotting predicted vs actual...')
    for i in range(n_SmartDevices):
        # Whole evaluation
        plt.figure(figsize=(20, 6))
        plt.suptitle(f'Device {i}: Predicted vs Actual')

        plt.subplot(2,1,1)
        plt.xlabel('Time Step')
        plt.ylabel('Actual')
        plt.scatter(range(len(devicesPredActual[i])), [x[1] for x in devicesPredActual[i]], s=3)

        plt.subplot(2,1,2)
        plt.xlabel('Time Step')
        plt.ylabel('Predicted')
        plt.plot(range(len(devicesPredActual[i])), [x[0] for x in devicesPredActual[i]])

        plt.savefig(f'{outFolder}/device{i}PredActual.png')

        # Zoomed
        plt.figure(figsize=(20, 6))
        plt.suptitle(f'Device {i}: Predicted vs Actual (First 1/4)')

        plt.subplot(2,1,1)
        plt.xlabel('Time Step')
        plt.ylabel('Actual')
        plt.scatter(range(n_zoomed_samples), [x[1] for x in devicesPredActual[i][:n_zoomed_samples]], s=3)

        plt.subplot(2,1,2)
        plt.xlabel('Time Step')
        plt.ylabel('Predicted')
        plt.plot(range(n_zoomed_samples), [x[0] for x in devicesPredActual[i][:n_zoomed_samples]])

        plt.savefig(f'{outFolder}/device{i}PredActual-Zoom.png')
        plt.close()

        # Zoomed 2
        state_change_index: int|None = None
        prevSample = devicesPredActual[i][0]
        for j, sample in enumerate(devicesPredActual[i]):
            if sample[1] != prevSample[1]:
                state_change_index = j
                break
            prevSample = sample
        del j

        if state_change_index is not None:
            start_index = state_change_index - (n_zoomed_state_change_samples // 2)
            start_index = 0 if start_index < 0 else start_index
            end_index = state_change_index + (n_zoomed_state_change_samples // 2)
            end_index = len(devicesPredActual[i]) if end_index >= len(devicesPredActual[i]) else end_index
            plt.figure(figsize=(20, 6))
            plt.suptitle(f'Device {i}: Predicted vs Actual (State change zoom)')

            plt.subplot(2,1,1)
            plt.axvline(x=state_change_index - start_index, color="red", linestyle="--", label=f"State change")
            plt.xlabel('Time Step')
            plt.ylabel('Actual')
            plt.scatter(range(n_zoomed_state_change_samples), [x[1] for x in devicesPredActual[i][start_index : end_index]], s=3)

            plt.subplot(2,1,2)
            plt.axvline(x=state_change_index - start_index, color="red", linestyle="--", label=f"State change")
            plt.xlabel('Time Step')
            plt.ylabel('Predicted')
            plt.plot(range(n_zoomed_state_change_samples), [x[0] for x in devicesPredActual[i][start_index : end_index]])

            plt.savefig(f'{outFolder}/device{i}PredActual-State-Change-Zoom.png')
            plt.close()

    print('Done')