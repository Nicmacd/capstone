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
id = torch.cuda.current_device() #returns you the ID of your current device
print("Device Name" + str(torch.cuda.get_device_name(id)))
print("Memory Allocated" + str(torch.cuda.memory_allocated(id))) #returns you the current GPU memory usage by tensors in bytes for a given device
print("Memory Reserved" + str(torch.cuda.memory_reserved(id)))
torch.cuda.empty_cache() 

def train(model, train_loader, criterion, optimizer, epochs, num_batch, batch_size):
    model.train()
    loss_array = np.zeros(epochs)
    #For each epoch
    for epoch in range(0, epochs):
        print("epoch", epoch)
        loss_sum = 0
        count = 0
        #For each batch
        for inputs, labels in train_loader:
            with torch.set_grad_enabled(True):
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            print("Batch [" + str(count) + "/" + str(num_batch) + "]: " + str(loss.item()))
            count = count + 1
            loss_numpy = (loss.cpu().detach().numpy())
            loss_sum = loss_sum + loss_numpy
        loss_array[epoch] = loss_sum

        if(epoch % 100 == 0):
            path = str("./model_") + str(epoch) + str(".pth")
            torch.save(model.state_dict(), path)

    # Plot the batch loss after each iteration
    plt.figure(figsize=(8, 5))
    plt.plot(loss_array, label='Batch Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Training Batch Loss ')
    plt.legend()
    plt.savefig("./loss.png")
    plt.show()

# Example data and targets
data_train = "../data/mfccData/train/images"
model_path = "./model.pth"

# Define transformations if needed
transform = transforms.Compose([transforms.ToTensor()])  # Example transformation

# Create a dataset
dataset_train = Marine_Mammal_Dataset(data_train)
epochs = 80
batch_size = 64
num_batch = int(len(dataset_train) / batch_size)

# Create DataLoader for training and testing sets
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# instantiations
model = CNN(dataset_train[0][0].shape)
model = model.cuda()

weights = torch.tensor([1.217, 7.913, 19.328])
criterion = nn.CrossEntropyLoss(weight=weights).cuda() #Should probably use some type of weighted cross entropy here due to classification task with class imbalance
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train(model, train_loader, criterion, optimizer, epochs, num_batch, batch_size)
torch.save(model.state_dict(), model_path)



