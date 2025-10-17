import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(x)
        x = self.conv2(x)

        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x




class CNN_Net_cifar(nn.Module):
    def __init__(self):
        super(CNN_Net_cifar, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1)

        self.conv1_1 = nn.Conv2d(64, 128, 1, 1)

        self.BN1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 64, 3, 1)

        self.conv3 = nn.Conv2d(64, 64, 1, 1)
        self.BN2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(256, 128)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)

        x = self.conv1_1(x)

        x = self.BN1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)

        return x

class CNN_Net_cifar_mnist(nn.Module):
    def __init__(self):
        super(CNN_Net_cifar_mnist, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv1_1 = nn.Conv2d(64, 128, 1, 1)

        self.BN1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 64, 3, 1)

        self.conv3 = nn.Conv2d(64, 64, 1, 1)
        self.BN2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 4)  # 10个数字，10类

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)

        x = self.conv1_1(x)

        x = self.BN1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)

        return x

class CNN_Net_SVHN(nn.Module):
    def __init__(self):
        super(CNN_Net_SVHN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv1_1 = nn.Conv2d(64, 128, 1, 1)

        self.BN1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 64, 3, 1)

        self.conv3 = nn.Conv2d(64, 64, 1, 1)
        self.BN2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.8)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)

        x = self.conv1_1(x)

        x = self.BN1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)

        return x

