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

# Example data and targets
data_test = "../data/melData/test/"
model_path = "./model.pth"

# Create the dataset
dataset_test = Marine_Mammal_Dataset(data_test)

# instantiations
model = CNN(dataset_test[0][0].shape)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

dolphins = 0
whales = 0 

Tp = 0
Fp = 0
Fn = 0
Tn = 0
correct = 0

for idx in range(len(dataset_test)):
    input = dataset_test[idx][0].unsqueeze(0)
    input = input.to(device)
    output = model(input)[0]
    value, indices = torch.max(output, 0)
    
    # TP
    if ((indices == 1) and (dataset_test[idx][1][1] == 1.)):
        Tp = Tp + 1
    # FP
    if ((indices == 1) and (dataset_test[idx][1][1] == 0.)):
        Fp = Fp + 1

    # FN
    if ((indices == 0) and (dataset_test[idx][1][1] == 1.)):
        Fn = Fn + 1

    # TN
    if ((indices == 0) and (dataset_test[idx][1][1] == 0.)):
        Tn = Tn + 1

    if (dataset_test[idx][1][indices] == 1.):
        correct = correct + 1

Accuracy = correct / len(dataset_test)
print('Accuracy')
print(Accuracy)
print('Tp')
print(Tp)
print('Fp')
print(Fp)
print('Fn')
print(Fn)
print('Tn')
print(Tn)
