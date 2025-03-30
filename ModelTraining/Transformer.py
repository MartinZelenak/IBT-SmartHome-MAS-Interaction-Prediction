import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
import Data

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position_encodings = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        position_encodings[:, 0::2] = torch.sin(position * div_term)
        position_encodings[:, 1::2] = torch.cos(position * div_term)
        position_encodings = position_encodings.unsqueeze(0).transpose(0, 1)
        self.register_buffer('position_encodings', position_encodings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.position_encodings[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, n_heads=4, num_layers=2, dropout=0.2) -> None:
        super(Transformer, self).__init__()

        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Create out folder
    outFolder = 'out/Transformer'
    os.makedirs(outFolder, exist_ok=True)
    print(f'Output folder: {outFolder}')
    

    # Hyper-parameters
    ## Dataset
    train_data_file_path = './datasets/year-5min.csv'
    eval_data_file_path = './datasets/month-5min.csv'
    n_total_time_columns = 6 # minute, hour, dayOfWeek, dayOfMonth, month, year
    n_SmartDevices = 8
    remove_columns = ['DayOfMonth', 'Month', 'Year'] # keep only minute, hour, dayOfWeek as time
    max_samples = None

    n_data_columns = n_total_time_columns + 1 + n_SmartDevices # timeColumns + location + deviceStates
    column_name_number_map = Data.get_column_name_number_map(train_data_file_path, 15)
    n_time_columns = n_total_time_columns - len(remove_columns)
    transform = transforms.Compose([Data.RemoveColumnsTransform(remove_feature_cols=[column_name_number_map[col_name] for col_name in remove_columns],  # Remove unused time columns
                                                                remove_label_cols=range(0, n_total_time_columns + 1))])                                 # Remove time columns and location
                                    # Data.MinuteHourToSlotNumberOfDayTransform(minute_column=column_name_number_map['Minute'], 
                                    #                                           hour_column=column_name_number_map['Hour'], 
                                    #                                           minutes_between_slots=5, 
                                    #                                           only_current_state=True)])
                                    # n_time_columns = n_time_columns - 1 # Remove SlotNumberOfDayReduction
    ## Model
    hidden_size = 256
    n_heads = 4
    num_layers = 3
    dropout = 0.2
    sequence_len = 6 # Time sequence length
    ## Training
    n_epochs = 1
    learning_rate = 0.001


    # Dataset
    train_dataset = Data.NextStateDataset(train_data_file_path, max_samples=max_samples, n_state_cols=n_data_columns, transform=transform, device=device)
    eval_dataset = Data.NextStateDataset(eval_data_file_path, max_samples=max_samples, n_state_cols=n_data_columns, transform=transform, device=device)
    n_feature_columns: int = train_dataset.sample_sizes()[0]

    # Model
    ## Model: Sequence of previous states -> next device states
    ## State: [timeColumns, location, deviceStates]
    model = Transformer(input_size=n_feature_columns,
                        output_size=n_SmartDevices,
                        d_model=hidden_size,
                        n_heads=n_heads,
                        num_layers=num_layers,
                        dropout=dropout).to(device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
            pred = model(sequence)[-1] # last sample prediction (next state)
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
    plt.title(f'Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(f'{outFolder}/loss.png')

    plt.figure(figsize=(20, 6))
    plt.scatter(range(len(sameStateLosses)), sameStateLosses, s=3)
    plt.title(f'Same State Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(f'{outFolder}/sameStateLoss.png')

    plt.figure(figsize=(20, 6))
    plt.scatter(range(len(stateChangeLosses)), stateChangeLosses, s=3)
    plt.title(f'State Change Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(f'{outFolder}/stateChangeLoss.png')

    # Save model
    print('Saving model...')
    torch.save(model.state_dict(), f'{outFolder}/{Transformer.__name__}.pth')
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
            pred: torch.Tensor = model(sequence)[-1] # last sample prediction

            # Save predicted and actual state
            for i in range(n_SmartDevices):
                devicesPredActual[i].append((float(pred[i]), float(nextState[i])))

            # Compare prediction with actual state
            pred = torch.round(pred)
            if torch.equal(pred, nextState):
                correct_guesses += 1
            if not torch.equal(currentState[n_time_columns+1:], nextState):
                total_state_changes += 1
                if torch.equal(pred, nextState):
                    correct_state_change_guesses += 1
            for i in range(n_SmartDevices):
                if not torch.equal(currentState[n_time_columns+1+i], nextState[i]):
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
    print('Plotting predicted vs actual...')
    for i in range(n_SmartDevices):
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
    
    print('Done')