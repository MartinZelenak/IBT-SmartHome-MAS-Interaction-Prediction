import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.modules.loss import _Loss
import plotext as plt
from typing import List, Tuple, Dict
import ActionsData

class DoActionLSTMNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super(DoActionLSTMNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class ActionClassLSTMNN(nn.Module):
    def __init__(self, input_size, n_devices, n_actions_per_device, hidden_size, num_layers):
        super(ActionClassLSTMNN, self).__init__()
        
        self.n_devices = n_devices
        self.n_actions_per_device = n_actions_per_device
        self.hidden_size = hidden_size
        self.output_size = n_devices * n_actions_per_device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
    
    def deviceActionToClass(self, device: int, action: int) -> int:
        '''Converts a device and action to a class number.\n
            class = (device-1) * n_actions_per_device + action
        '''
        class_ = (device - 1) * self.n_actions_per_device + (action - 1) # device and action 1-indexed
        if class_ < 0 or class_ > self.output_size:
            raise ValueError(f'Invalid device-action pair: device={device}, action={action}')
        return class_
    
    def classToDeviceAction(self, class_: int) -> Tuple[int, int]:
        '''Converts a class number to a device and action.\n
            device = class // n_actions_per_device + 1 and action = class % n_actions_per_device + 1
        '''
        device = class_ // self.n_actions_per_device + 1
        action = class_ % self.n_actions_per_device + 1
        return (device, action)

def findMinimalLoss(criterion: _Loss, pred: torch.Tensor, labels: List[torch.Tensor]) -> torch.Tensor:
    loss = criterion(pred, labels[0])
    for label in labels[1:]:
        loss = min(loss, criterion(pred, label))
    return loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Hyper-parameters
    n_SmartDevices = 8
    max_samples = None
    n_time_columns = 3 # minute, hour, day
    n_total_sample_columns = n_time_columns + 1 + n_SmartDevices # timeColumns + location + deviceStates
    transform = transforms.Compose([
        ActionsData.LessTimeColumnsTransform(n_time_columns, n_total_time_columns=5)])
    hidden_size = 64
    num_layers = 2
    sequence_len = 6 # Time sequence length
    n_epochs = 10
    learning_rate = 0.001

    # Dataset
    dataset = ActionsData.ActionsDataset('datasets/month-5min.csv', max_samples=max_samples, n_smart_devices=n_SmartDevices, transform=transform, device=device)

    # Models
    ## doActionModel: [minute, hour, day, location, deviceState * n_SmartDevices...] -> <0,1>
    doActionModel = DoActionLSTMNN(input_size=n_total_sample_columns, 
                                   hidden_size=hidden_size, 
                                   num_layers=num_layers).to(device)
    ## actionClassModel: [minute, hour, day, month, year, location, deviceState * n_SmartDevices...] -> device_action_class
    ## device_action_classes: n_SmartDevices * (ON/OFF) + no_action
    actionClassModel = ActionClassLSTMNN(input_size=n_total_sample_columns, 
                   n_devices=n_SmartDevices, 
                   n_actions_per_device=2, 
                   hidden_size=hidden_size, 
                   num_layers=num_layers).to(device)

    # Loss and optimizer
    ## doActionModel
    doActionCriterion = nn.BCELoss()
    doActionOptimizer = torch.optim.Adam(doActionModel.parameters(), lr=learning_rate)
    ## actionClassModel
    # actionClassCriterion = nn.CrossEntropyLoss(weight=torch.tensor([0.01] + [1000] * (n_SmartDevices*2)).to(device))ˇ
    actionClassCriterion = nn.CrossEntropyLoss()
    actionClassOptimizer = torch.optim.Adam(actionClassModel.parameters(), lr=learning_rate)

    # Training loop
    doActionModel.train()
    actionClassModel.train()
    doActionLosses: List[float] = []
    actionLosses: Dict[int, List[float]] = {}
    n_noAction_since_last_action: int = 0
    for epoch in range(n_epochs):
        sequence: torch.Tensor = torch.Tensor(size=(0,n_total_sample_columns)).to(device)
        for i, (features, actions) in enumerate(dataset): # type: ignore
            sequence = torch.cat((sequence, features.unsqueeze(0)), 0)
            if sequence.size(0) >= sequence_len:
                sequence = sequence[-sequence_len:] # keep only last sequence_len samples

            labels: List[torch.Tensor] = [torch.tensor(actionClassModel.deviceActionToClass(action[0].item(), action[1].item()), dtype=torch.long).to(device) for action in actions] # list of all classes

            # Train doActionModel
            pred = doActionModel(sequence)[-1] # last sample prediction
            doActionLoss = doActionCriterion(pred, torch.tensor([1.0], device=device) if len(labels) != 0 else torch.tensor([0.0], device=device))
            
            ## Backward and optimize
            doActionOptimizer.zero_grad()
            doActionLoss.backward()
            doActionOptimizer.step()

            ## Record losses
            doActionLosses.append(doActionLoss.item())

            # Train actionClassModel
            if len(labels) != 0:
                pred = actionClassModel(sequence)[-1] # last sample prediction
                actionClassLoss = findMinimalLoss(actionClassCriterion, pred, labels)

                ## Backward and optimize
                actionClassOptimizer.zero_grad()
                actionClassLoss.backward()
                actionClassOptimizer.step()

                ## Record losses
                actualClass = actionClassModel.deviceActionToClass(actions[0][0].item(), actions[0][1].item())
                if actualClass not in actionLosses.keys():
                    actionLosses[actualClass] = []
                actionLosses[actualClass].append(actionClassLoss.item())

            # Print progress
            # if len(labels) != 0:
            #     print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(dataset)}], Loss: {loss.item():.4f} ; {len(labels)} actions')
            # elif ((i+1)) % 100 == 0:
            if ((i+1)) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(dataset)}], DoActionLoss: {doActionLoss.item():.4f}, ActionClassLoss: {actionClassLoss.item():.4f}')

    # Evaluate
    dataset = ActionsData.ActionsDataset('datasets/week-5min.csv', n_smart_devices=n_SmartDevices, transform=transform, device=device)
    doActionModel.eval()
    actionClassModel.eval()
    with torch.no_grad():
        sequence: torch.Tensor = torch.empty((0,n_total_sample_columns), device=device)
        total_samples_with_actions = 0
        correct_doAction_action_guesses = 0
        correct_doAction_guesses = 0
        correct_doAction_guesses_weighted = 0
        total_actionClass_guesses = 0
        correct_actionClass_guesses = 0
        correct_actionClass_guesses_weighted = 0
        for i, (features, actions) in enumerate(dataset): # type: ignore
            sequence = torch.cat((sequence, features.unsqueeze(0)), 0)
            if sequence.size(0) >= sequence_len:
                sequence = sequence[-sequence_len:] # keep only last sequence_len samples

            # DoActionModel
            pred = doActionModel(sequence)[-1] # last sample prediction
            doAction = 1 if pred.item() > 0.5 else 0
            if len(actions) == 0:
                if doAction == 0:
                    correct_doAction_guesses += 1
                    correct_doAction_guesses_weighted += pred.item()
            else:
                total_samples_with_actions += 1
                if doAction == 1:
                    correct_doAction_guesses += 1
                    correct_doAction_action_guesses += 1
                    correct_doAction_guesses_weighted += pred.item()

            # ActionClassModel
            if len(actions) != 0:
                pred = actionClassModel(sequence)[-1] # last sample prediction
                class_ = int(torch.argmax(pred).item())
                device, action = actionClassModel.classToDeviceAction(class_)
                percentages = F.softmax(pred, dim=0)

                total_actionClass_guesses += 1
                for label in actions:
                    if label[0].item() == device and label[1].item() == action:
                        correct_actionClass_guesses += 1
                        correct_actionClass_guesses_weighted += percentages[class_]
                        break

        print(f'DoActionModel: {correct_doAction_guesses}/{len(dataset)} ({(correct_doAction_guesses/len(dataset))*100:.2f}%)')
        print(f'DoActionModel (only actions): {correct_doAction_action_guesses}/{total_samples_with_actions} ({(correct_doAction_action_guesses/total_samples_with_actions)*100:.2f}%)')
        print(f'DoActionModel (weighted): {correct_doAction_guesses_weighted}/{len(dataset)} ({(correct_doAction_guesses_weighted/len(dataset))*100:.2f}%)')
        print(f'ActionClassModel: {correct_actionClass_guesses}/{total_actionClass_guesses} ({(correct_actionClass_guesses/total_actionClass_guesses)*100:.2f}%)')
        print(f'ActionClassModel (weighted): {correct_actionClass_guesses_weighted}/{total_actionClass_guesses} ({(correct_actionClass_guesses_weighted/total_actionClass_guesses)*100:.2f}%)')
    
    # Try to predict
    with torch.no_grad():
        sequence: torch.Tensor = torch.empty((0, n_total_sample_columns), device=device)
        preds: int = 0
        for i, (features, actions) in enumerate(dataset): # type: ignore
            sequence = torch.cat((sequence, features.unsqueeze(0)), 0)
            if sequence.size(0) >= sequence_len:
                sequence = sequence[-sequence_len:] # keep only last sequence_len samples
            if len(actions) == 0:
                continue
            pred = actionClassModel(sequence)[-1] # last sample prediction
            device, action = actionClassModel.classToDeviceAction(int(torch.argmax(pred).item()))
            percentages = F.softmax(pred, dim=0)
            print(f'Prediction: device={device}, action={action}, confidance={torch.max(percentages)} ; Actual: devices={actions[:,0]}, actions={actions[:,1]}')
            print('Prediction output:')
            print(percentages)

            preds += 1
            if preds >= 10:
                break

    # Plot loss
    # plt.scatter(range(len(noActionLosses)), noActionLosses, label='No Action', color='blue')
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'brown']
    for i, key in enumerate(actionLosses):
        plt.clear_figure()
        plt.scatter(range(len(actionLosses[key])), actionLosses[key], color=colors[i % len(colors)])
        plt.title(f'Device {actionClassModel.classToDeviceAction(key)[0]} Action {actionClassModel.classToDeviceAction(key)[1]} Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.show()

    plt.clear_figure()
    plt.scatter(range(len(doActionLosses)), doActionLosses)
    plt.title(f'No Action Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

    plt.clear_figure()
    plt.bar(range(len(actionLosses)), [losses[-1] for losses in actionLosses.values()], label=f'Device {actionClassModel.classToDeviceAction(key)[0]} Action {actionClassModel.classToDeviceAction(key)[1]}')
    plt.title('Final losses for each action class')
    plt.xlabel('Action')
    plt.ylabel('Loss')
    plt.show()