import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
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
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.sigmoid(x).squeeze(1)
    
    
# class CNN_features(nn.Module):
#     def __init__(self):
#         super(CNN_features, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
#         self.dropout1 = nn.Dropout(0.4)
#         self.dropout2 = nn.Dropout(0.4)
#         self.dropout3 = nn.Dropout(0.4)
#         self.fc1 = nn.Linear(64 * 2 * 15, 26)
#         self.fc2 = nn.Linear(26*2, 16)
#         self.fc3 = nn.Linear(16, 1)

#     def forward(self, x, features):
#         x = self.pool(nn.functional.leaky_relu(self.conv1(x)))
#         x = self.dropout1(x)
#         x = self.pool(nn.functional.leaky_relu(self.conv2(x)))
#         x = self.dropout2(x)
#         x = self.pool(nn.functional.leaky_relu(self.conv3(x)))
#         x = self.dropout3(x)
#         x = torch.flatten(x, 1)
#         x = nn.functional.relu(self.fc1(x))
#         end = torch.cat((x, features.squeeze(1)), 1)
#         x = nn.functional.relu(self.fc2(end))
#         x = self.fc3(x)
#         return nn.functional.sigmoid(x).squeeze(1)
    
    
    
class CNN_features(nn.Module):
    def __init__(self):
        super(CNN_features, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(64 * 2 * 15, 26)
        self.fc2 = nn.Linear(26*2, 1)

    def forward(self, x, features):
        x = self.pool(nn.functional.leaky_relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(nn.functional.leaky_relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(nn.functional.leaky_relu(self.conv3(x)))
        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        end = torch.cat((x, features.squeeze(1)), 1)
        x = self.fc2(end)
        return nn.functional.sigmoid(x).squeeze(1)
    
    
class noDIA_features(nn.Module):
    def __init__(self, n_feat=38):
        super(noDIA_features, self).__init__()
        self.linear1 = nn.Linear(n_feat, 16)
        # self.linear2 = nn.Linear(32, 16)
        self.dropout1 = nn.Dropout(0.4)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        # x = nn.functional.relu(self.linear2(x))
        x = self.fc(x)
        return nn.functional.sigmoid(x).squeeze(1)
    
class noDIA_CNN_features(nn.Module):
    def __init__(self, input_shape=(1, 51, 102), n_feat=38):
        super(noDIA_CNN_features, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 1, kernel_size=7, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(32 * 2 * 20, 32)
        self.fc2 = nn.Linear(32, 16)
        self.linearf = nn.Linear(n_feat, 16)
        self.fc = nn.Linear(16*2, 1)

    def forward(self, x, features):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        feat = nn.functional.relu(self.linearf(features))
        end = torch.cat((x, feat), 1)
        x = self.fc(end)
        return nn.functional.sigmoid(x).squeeze(1)

class noDIA_CNN(nn.Module):
    def __init__(self, input_shape=(1, 51, 102),):
        super(noDIA_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 1, kernel_size=7, stride=1)
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