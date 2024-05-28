import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(32)

        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * 28 * 28, 64)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)

        # Output Layer
        # TODO change to 4
        self.fc2 = nn.Linear(64, 12)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batchnorm2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.batchnorm3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)