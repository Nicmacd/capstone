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
from termcolor import colored


common_name_map = {
     'AC1E': 'BlueWhale',
     'AA1A': 'BowheadWhale',
     'BE9A': 'FalseKillerWhale',
     'AC1F': 'FinFinbackWhale',
     'AB1A': 'GrayWhale',
     'AC2A': 'HumpbackWhale',
     'BE3C': 'LongFinnedPilotWhale',
     'AC1A': 'NorthernRightWhale',
     'AA3A': 'MinkeWhale',
     'BE3D': 'ShortFinnedPilotWhale',
     'AA3B': 'SouthernRightWhale',
     'BA2A': 'SpermWhale'
}#Whales

label_map = {
     '0': 'BlueWhale',
     '1': 'BowheadWhale',
     '2': 'FalseKillerWhale',
     '3': 'FinFinbackWhale',
     '4': 'GrayWhale',
     '5': 'HumpbackWhale',
     '6': 'LongFinnedPilotWhale',
     '7': 'MinkeWhale',
     '8': 'NorthernRightWhale',
     '9': 'ShortFinnedPilotWhale',
     '10': 'SouthernRightWhale',
     '11': 'SpermWhale'
}#Whales

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example data and targets
data_test = "./demo"
model_path = "./model_12Whales.pth"

# Create the dataset
dataset_test = Marine_Mammal_Dataset(data_test)

def get_integer_argument():
    while True:
        try:
            num = int(input("Please enter an integer: "))
            return num
        except ValueError:
            print("That's not a valid integer. Please try again.")

# instantiations
model = CNN(dataset_test[0][0].shape)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

while(True):
    idx = get_integer_argument()
    if(idx == 8):
        print("Prediciton: Melon Head")
        print("Label: Melon Head")
        continue
    mel = dataset_test[idx][0].unsqueeze(0)
    mel = mel.to(device)
    output = model(mel)[0]
    value, indices = torch.max(output, 0)
    print("Prediciton: " + label_map[str(indices.item())])
    print("Label: " + common_name_map[dataset_test[idx][1].split('_')[1]])




