import numpy as np
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from dataset import Marine_Mammal_Dataset
from model import CNN
from torchinfo import summary

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, criterion, optimizer, epochs, num_batch):
    model.train()
    loss_array = np.zeros(epochs)
    for epoch in range(0, epochs):
        print("epoch", epoch)
        loss_sum = 0
        count = 0
        for inputs, labels in train_loader:
            inputs =  inputs.to(device)
            labels =  labels.to(device)
            with torch.set_grad_enabled(True):

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            print("Batch [" + str(count) + "/" + str(num_batch) + "]: " + str(loss.item()))
            count = count + 1
            loss_numpy = loss.cpu().detach().numpy()
            loss_sum = loss_sum + loss_numpy
        loss_array[epoch] = loss_sum

    # Plot the batch loss after each iteration
    plt.figure(figsize=(8, 5))
    plt.plot(loss_array, label='Batch Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Training Batch Loss ')
    plt.legend()
    plt.show()

# Example data and targets
data_train = "../data/mfccData/train/"
model_path = "./model.pth"

# Define transformations if needed
transform = transforms.Compose([transforms.ToTensor()])  # Example transformation

# Create a dataset
dataset_train = Marine_Mammal_Dataset(data_train)
epochs = 5
batch_size = 32
num_batch = int(len(dataset_train) / batch_size)

# Create DataLoader for training and testing sets
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# instantiations
model = CNN(dataset_train[0][0].shape)
model = model.to(device)

criterion = nn.MSELoss() #Should probably use some type of weighted cross entropy here due to classification task with class imbalance
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, criterion, optimizer, epochs, num_batch)
torch.save(model.state_dict(), model_path)



