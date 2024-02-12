import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from CNN.model import CNN
def train():
    for epoch in range(0, epochs):
        print("epoch", epochs)

        for batch in range(0, num_batch):

            for inputs, labels in train_loader:
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                print('Batch Loss:', loss.item())

            # Plot the batch loss after each iteration
            plt.figure(figsize=(8, 5))
            plt.plot(loss, label='Batch Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'Training Batch Loss ')
            plt.legend()
            plt.show()

# Define your dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

# Example data and targets
data_train = "data/mfccData/train"
data_train_labels = "data/mfccData/labels"

# Define transformations if needed
transform = transforms.Compose([transforms.ToTensor()])  # Example transformation

# Create a dataset
dataset_train = CustomDataset(data_train, transform=transform)

dataset = dataset_train.transform

epochs = 10
batch_size = 16
num_batch = int(len(dataset) / batch_size)

# Create DataLoader for training and testing sets
train_loader = DataLoader(dataset_train, data_train_labels, batch_size=batch_size, shuffle=True)

# instantiations
model = CNN.model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


