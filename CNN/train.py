import numpy as np
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from CNN.model import CNN

mfcc_train_imgs_folder = "./data/mfccData/train/images/"
mfcc_train_lbls_folder = "./data/mfccData/train/labels/"

def train(model, train_loader, criterion, optimizer, epoch, num_batch):
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
    def __init__(self, images_folder, labels_folder, transform=None):
        self.images_folder = mfcc_train_imgs_folder
        self.labels_folder = mfcc_train_lbls_folder
        self.transform = transform
        self.image_filenames = os.listdir(mfcc_train_imgs_folder)
        self.label_filenames = os.listdir(mfcc_train_lbls_folder)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.image_filenames[idx])
        label_path = os.path.join(self.labels_folder, self.label_filenames[idx])

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Read label from corresponding text file
        with open(label_path, 'r') as file:
            label = int(file.read().strip())

        return image, label


# Example data and targets
data_train = "data/mfccData/train/images"
data_train_labels = "data/mfccData/train/labels"

# Define transformations if needed
transform = transforms.Compose([transforms.ToTensor()])  # Example transformation

# Create a dataset
dataset_train = CustomDataset(data_train, data_train_labels, transform=transform)

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

train(model, train_loader, criterion, optimizer, epochs, num_batch)


