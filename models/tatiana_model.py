import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape=(1, 51, 153)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(64 * 2 * 15, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.sigmoid(x).squeeze(1)
    
    
class noDIA_CNN(nn.Module):
    def __init__(self):
        super(noDIA_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(32 * 2 * 20, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.sigmoid(x).squeeze(1)
