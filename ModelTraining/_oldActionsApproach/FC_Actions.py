import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import ActionsData

class FCNN(nn.Module):
    def __init__(self, input_size, device_classes, action_classes, hidden_size=15):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.device_fc = nn.Linear(hidden_size, device_classes)
        self.action_fc = nn.Linear(hidden_size, action_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        output = self.device_fc(x), self.action_fc(x)
        # output = F.relu(y), F.relu(z)
        return output
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Input Data
    n_total_time_columns = 5 # minute, hour, day, month, year
    n_SmartDevices = 8
    n_DeviceActions = 2
    train_data_file = 'datasets/workday-5min-100iters.csv'
    test_data_file = 'datasets/workday-5min-6iters.csv'

    # Hyper-parameters
    max_samples = None
    n_time_columns = 5 # use n_time_columns out of n_total_time_columns
    transform = transforms.Compose([
        ActionsData.LessTimeColumnsTransform(n_time_columns, n_total_time_columns=5), 
        ActionsData.NoActionsAsZeroZeroActionTransform(), 
        ActionsData.OnlyFirstActionTransform()])
    only_samples_with_actions = True
    hidden_size = 100
    n_epochs = 20
    batch_size = 5 # > 1 works only when ( NoActionsAsZeroZeroActionTransform() or dataset(only_samples_with_actions=True) ) and OnlyFirstActionTransform() are used
    learning_rate = 0.001

    # Dataset
    dataset = ActionsData.ActionsDataset(train_data_file, max_samples=max_samples, n_smart_devices=n_SmartDevices, only_samples_with_actions=only_samples_with_actions, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    # Model: [time_column * n_time_columns..., location, deviceState * n_SmartDevices...] -> [[device_class_logit], [action_class_logit]]
    model = FCNN(n_time_columns + 1 + n_SmartDevices, 
                 n_SmartDevices + 1,  # +1 for the "no device" class (0)
                 n_DeviceActions + 1, # +1 for the "no action" class (0)
                 hidden_size=hidden_size
                ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        for i, (features, labels) in enumerate(dataloader):
            # Forward pass
            features: torch.Tensor = features.to(device)
            labels: torch.Tensor = labels[:, 0].to(device) # unpack first action from the "list" of actions for each sample
            devices_pred, actions_pred = model(features)

            device_loss = criterion(devices_pred, labels[:, 0].long())
            action_loss = criterion(actions_pred, labels[:, 1].long())
            loss = device_loss + action_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ((i+1)) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Evaluate and try predicting
    with torch.no_grad():
        test_dataset = ActionsData.ActionsDataset(test_data_file, max_samples=None, n_smart_devices=n_SmartDevices, only_samples_with_actions=only_samples_with_actions, transform=transform)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

        # Evaluate
        n_devices_correct = 0
        n_actions_correct = 0
        n_total = 0
        for features, labels in test_dataloader:
            features = features.to(device)
            labels = labels[:, 0].to(device) # unpack one label from the "list" of labels for each sample
            
            devices_pred_logits, actions_pred_logits = model(features)
            _, devices_pred = torch.max(devices_pred_logits, dim=1)
            _, actions_pred = torch.max(actions_pred_logits, dim=1)

            device_errors = torch.abs(devices_pred - labels[:, 0])
            action_errors = torch.abs(actions_pred - labels[:, 1])

            n_total += batch_size
            for i in range(len(device_errors)):
                if device_errors[i].item() == 0:
                    n_devices_correct += 1
                if action_errors[i].item() == 0:
                    n_actions_correct += 1

        # Try predicting
        for i, (features, labels) in enumerate(test_dataloader):
            features = features.to(device)
            labels = labels[:, 0] # unpack one label from the "list" of labels for each sample

            devices_pred_logits, actions_pred_logits = model(features)
            _, devices_pred = torch.max(devices_pred_logits, dim=1)
            _, actions_pred = torch.max(actions_pred_logits, dim=1)

            print(f'Guess {i+1}:')
            print(f'- Features: {features.to("cpu").numpy()}')
            print(f'- Labels: {labels.numpy()}')
            print(f'- Predictions: {torch.cat((devices_pred, actions_pred)).to("cpu").numpy()}')
            print()
            if i+1 == 10:
                break

        print(f'Device accuracy: {n_devices_correct/n_total} [{n_devices_correct}/{n_total}]')
        print(f'Action accuracy: {n_actions_correct/n_total} [{n_actions_correct}/{n_total}]')