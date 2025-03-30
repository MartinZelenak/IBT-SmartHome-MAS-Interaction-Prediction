import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from typing import Optional

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

class ActionsDataset(Dataset):
    '''Dataset of inhabitants' actions and states.
    Each sample is a tuple of (state, actions).
    State tensor of [minute, hour, day, month, year, location, devices_states...].
    Actions are tensor of [total_actions * [device_number, action_number]] of shape [total_actions, 2].'''
    def __init__(self, csv_file: str, n_smart_devices: int, max_samples: Optional[int]=None, only_samples_with_actions: bool=False, transform=None, device=None):
        self.features: np.ndarray = np.loadtxt(csv_file, 
                                    delimiter=',',
                                    skiprows=1, 
                                    dtype='i4',
                                    usecols=range(0, 6+n_smart_devices), # ['minute', 'hour', 'day', 'month', 'year', 'location'] + [f'device{i}' for i in range(0,n_smart_devices)]
                                    max_rows=max_samples)
        
        # TODO: Load actions more efficiently
        labels = np.loadtxt(csv_file,
                                    delimiter=',',
                                    skiprows=1,
                                    dtype=[('actions', 'O')],
                                    usecols=-1,
                                    max_rows=max_samples,
                                    converters={-1: lambda s: actionsArrayStringToActionsList(s.decode())})
        self.labels: np.ndarray = np.array([label[0] for label in labels], dtype='O') # Remove the extra dimension

        self.n_samples = self.features.shape[0]
        self.transform = transform
        self.device = device

        if only_samples_with_actions:
            self._filter_samples_with_actions()

    def _filter_samples_with_actions(self):
        to_include = [len(label) > 0 for label in self.labels]
        self.features = self.features[to_include]
        self.labels = self.labels[to_include]
        if len(self.features) == 0:
            raise ValueError('No samples with actions found in given CSV file.')
        self.n_samples = self.features.shape[0]
        

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        sample = (self.features[index], self.labels[index])

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            if sample is None:
                return None

        sample = (torch.tensor(sample[0], dtype=torch.float32, device=self.device), torch.tensor(sample[1], dtype=torch.float32, device=self.device))
        
        return sample

class LessTimeColumnsTransform:
    '''Keeps only the first n_time_columns time columns.'''
    def __init__(self, n_time_columns: int, n_total_time_columns: int = 5):
        if n_time_columns > n_total_time_columns:
            raise ValueError('n_time_columns must be less than or equal to n_total_time_columns.')
        self.n_time_columns = n_time_columns
        self.n_total_time_columns = n_total_time_columns
    
    def __call__(self, sample):
        if self.n_time_columns == self.n_total_time_columns:
            return sample
        sample = (np.concatenate((sample[0][:self.n_time_columns], sample[0][self.n_total_time_columns:])), sample[1])
        return sample
    
class NoActionsAsZeroZeroActionTransform:
    '''Transforms samples with no actions to samples with one action (0, 0).'''
    def __call__(self, sample):
        if len(sample[1]) == 0:
            sample = (sample[0], [(0, 0)])
        return sample
    
class OnlyFirstActionTransform:
    '''Transforms samples with multiple actions to samples with only the first action.'''
    def __call__(self, sample):
        if len(sample[1]) > 1:
            sample = (sample[0], sample[1][0:1][:])
        return sample

if __name__ == "__main__":
    # transform = OnlyFirstActionTransform()
    transform = transforms.Compose([LessTimeColumnsTransform(3,5), NoActionsAsZeroZeroActionTransform()])
    dataset = ActionsDataset('./datasets/workday-5min.csv', n_smart_devices=8, max_samples=None, only_samples_with_actions=True,transform=transform)
    print(f'Number of samples: {len(dataset)}')

    # Dataset test
    index = 73 if len(dataset) > 73 else 0
    print(dataset[index])

    # Dataloader test
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    for i, (features, labels) in enumerate(dataloader):
        print(f'Batch {i}:')
        print(features)
        print(labels)
        print()
        if i == 0:
            break