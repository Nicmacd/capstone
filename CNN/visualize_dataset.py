import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dataset import Marine_Mammal_Dataset

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example data and targets
data_test = "../data/melData/train/"

# Create the dataset
dataset_test = Marine_Mammal_Dataset(data_test)

dolphins = 0
orcas = 0 
total = 0 

for idx in range(len(dataset_test)):
    if(dataset_test[idx][1][0] == 1.):
        orcas = orcas + 1

    if(dataset_test[idx][1][1] == 1.):
        dolphins = dolphins + 1

    total = total + 1


print("Doplphins:")
print(dolphins)
print("Orcas:")
print(orcas)
print("Total:")
print(total)