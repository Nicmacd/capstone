import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dataset import Marine_Mammal_Dataset

def label_mapping(label):
    species = label.split("_")[0]
    if species == "whale":
        return torch.tensor([1, 0, 0], dtype=torch.float32)
    elif species == "dolphin":
        return torch.tensor([0, 1, 0], dtype=torch.float32)
    elif species == "seal":
        return torch.tensor([0, 0, 1], dtype=torch.float32)
    else:
        print("Label not found")
        return None

# Example data and targets
img_dir = "../data/melData/train/"

labels = []
# Create the dataset
for filename in os.listdir(img_dir):
        # Check if the item in the folder is a file (not a directory)
        if os.path.isfile(os.path.join(img_dir, filename)):
            labels.append(filename)


labels = torch.stack([label_mapping(label) for label in labels])

weightTensor = []

dolphins = 0
whales = 0 
seals = 0
total = 0 

for idx in range(len(labels)):
    if(labels[idx][0] == 1.):
        whales = whales + 1

    if(labels[idx][1] == 1.):
        dolphins = dolphins + 1


    if(labels[idx][2] == 1.):
        seals = seals + 1

    total = total + 1

weightTensor.append(round(len(labels)/whales,3))
weightTensor.append(round(len(labels)/dolphins,3))
weightTensor.append(round(len(labels)/seals,3))


print("Whales:")
print(whales)
print("Doplphins:")
print(dolphins)
print("Seals:")
print(seals)
print("Total:")
print(total)
print("Weights:")
print(weightTensor)