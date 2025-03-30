from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# TODO: Replace NextStateDataset with SequentialDataset in all experiments and delete this!

def actionsArrayStringToActionsList(string: str):
    if string[0] == '[' and string[-1] == ']':
        string = string[1:-1]
    
    if ':' not in string:
        return []
    
    actionStringsList = string.split(';')
    actionsList = []
    for actionString in actionStringsList:
        action = actionString.split(':')
        actionsList.append((int(action[0]), int(action[1])))
    return np.array(actionsList)

class NextStateDataset(Dataset):
    '''Dataset for the next state prediction problem.
        The dataset is loaded from a csv file and the samples are pairs of consecutive rows.
        The samples are returned as tuples of the current and next state.'''
    def __init__(
        self, csv_file: str, n_state_cols: int, max_samples: Optional[int] = None, transform=None, device=None
    ):
        column_names: np.ndarray = np.loadtxt(csv_file, delimiter=',', max_rows=1, dtype=str)[:n_state_cols]
        self.column_name_number_map: Dict[str, int] = {name: i for i, name in enumerate(column_names)}

        self.states: np.ndarray = np.loadtxt(csv_file, 
                                    delimiter=',',
                                    skiprows=1, 
                                    dtype='i4',
                                    usecols=range(0, n_state_cols),
                                    max_rows=max_samples)
        self.n_samples = self.states.shape[0]
        self.transform = transform
        self.device = device

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if index >= self.n_samples:
            raise IndexError('Index out of bounds.')

        sample = (self.states[index], self.states[index+1] if index+1 < self.n_samples else self.states[index])

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            if sample is None:
                return None

        sample = (torch.tensor(sample[0], dtype=torch.float32, device=self.device), torch.tensor(sample[1], dtype=torch.float32, device=self.device))

        return sample

    def sample_sizes(self):
        sample = self[0]
        if sample is None:
            raise ValueError('Dataset has no samples.')
        return (sample[0].shape[0], sample[1].shape[0])


class RemoveColumnsTransform:
    '''Removes the specified columns from samples.
        if remove_label_cols is None, it removes the same columns from both features and labels.'''
    def __init__(self, remove_feature_cols: Sequence[int], remove_label_cols: Optional[Sequence[int]]=None):
        self.remove_feature_cols = remove_feature_cols
        self.remove_label_cols = remove_label_cols if remove_label_cols else remove_feature_cols
    
    def __call__(self, sample):
        sample = (np.delete(sample[0], self.remove_feature_cols), np.delete(sample[1], self.remove_label_cols))
        return sample

class MinuteHourToSlotNumberOfDayTransform:
    '''Converts the minute and hour columns to a single slot number of the day column.
        The slot number is calculated as (hour * 60 + minute) // minutes_between_slots.
        The minute column is replaced by the slot number and the hour column is removed.'''
    def __init__(self, minute_column: int, hour_column: int, minutes_between_slots: int, only_current_state: bool = False) -> None:
        self.minute_column = minute_column
        self.hour_column = hour_column
        self.minutes_between_slots = minutes_between_slots
        self.only_current_state = only_current_state

    def __call__(self, sample):
        current_state = sample[0]
        current_state[self.minute_column] = (current_state[self.minute_column] + current_state[self.hour_column] * 60) // self.minutes_between_slots
        current_state = np.delete(current_state, self.hour_column)

        if self.only_current_state:
            return (current_state, sample[1])
        
        next_state = sample[1]
        next_state[self.minute_column] = (next_state[self.minute_column] + next_state[self.hour_column] * 60) // self.minutes_between_slots
        next_state = np.delete(next_state, self.hour_column)
        return (current_state, next_state)

def get_column_name_number_map(csv_file: str, n_state_cols: int) -> Dict[str, int]:
    column_names: np.ndarray = np.loadtxt(csv_file, delimiter=',', max_rows=1, dtype=str)[:n_state_cols]
    return {name: i for i, name in enumerate(column_names)}

if __name__ == "__main__":
    file_path = './datasets/month-5min.csv'
    n_state_cols = 15
    column_name_number_map = get_column_name_number_map(file_path, n_state_cols)

    remove_time_columns = [column_name_number_map[col_name] for col_name in ['DayOfMonth', 'Month', 'Year']] # keep only minute, hour, dayOfWeek
    n_total_time_columns = 6 # minute, hour, dayOfWeek, dayOfMonth, month, year
    transform = transforms.Compose([RemoveColumnsTransform(remove_feature_cols=remove_time_columns,                # Remove unused time columns
                                                                remove_label_cols=range(0, n_total_time_columns + 1)),  # Remove time columns and location
                                    MinuteHourToSlotNumberOfDayTransform(minute_column=0, hour_column=1, minutes_between_slots=5, only_current_state=True)])

    dataset = NextStateDataset(file_path, n_state_cols=n_state_cols, max_samples=None, transform=transform, device=None)
    print(f'Number of samples: {len(dataset)}')
    print(f'Sample sizes: {dataset.sample_sizes()}')
    print(f'Column name number map: {dataset.column_name_number_map}')

    # Dataset test
    index = 2500
    print(dataset[index])
    print(dataset[index+1])
    for i, (current, next) in enumerate(dataset): # type: ignore
        print(f'Sample {i}:')
        print(current)
        print(next)
        print()
        if i == 1:
            break


    # Dataloader test
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    for i, (features, labels) in enumerate(dataloader):
        print(f'Batch {i}:')
        print(features)
        print(labels)
        print()
        if i == 0:
            break
