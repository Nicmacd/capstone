import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dataset import Marine_Mammal_Dataset

common_name_map = {
     'BB1A': 'BelugaWhiteWhale',
     'AC1E': 'BlueWhale',
     'AA1A': 'BowheadWhale',
     'BE9A': 'FalseKillerWhale',
     'AC1F': 'FinFinbackWhale',
     'AB1A': 'GrayWhale',
     'AC2A': 'HumpbackWhale',
     'BE3C': 'LongFinnedPilotWhale',
     'BD10A': 'MelonHeadedWhale',
     'AC1A': 'MinkeWhale',
     'AA3A': 'NorthernRightWhale',
     'BE3D': 'ShortFinnedPilotWhale',
     'AA3B': 'SouthernRightWhale',
     'BA2A': 'SpermWhale'
}#Whales

def label_mapping(label):
    species_code = label.split("_")[1]
    label_tensor = torch.zeros(12, dtype=torch.float32)
    if species_code == 'AC1E':
        label_tensor[0] = 1  # BlueWhale
    elif species_code == 'AA1A':
        label_tensor[1] = 1  # BowheadWhale
    elif species_code == 'BE9A':
        label_tensor[2] = 1  # FalseKillerWhale
    elif species_code == 'AC1F':
        label_tensor[3] = 1  # FinFinbackWhale
    elif species_code == 'AB1A':
        label_tensor[4] = 1  # GrayWhale
    elif species_code == 'AC2A':
        label_tensor[5] = 1  # HumpbackWhale
    elif species_code == 'BE3C':
        label_tensor[6] = 1  # LongFinnedPilotWhale
    elif species_code == 'BD10A':
        label_tensor[7] = 1  # MelonHeadedWhale
    elif species_code == 'AC1A':
        label_tensor[8] = 1  # MinkeWhale
    elif species_code == 'BE3D':
        label_tensor[9] = 1  # ShortFinnedPilotWhale
    elif species_code == 'AA3B':
        label_tensor[10] = 1  # SouthernRightWhale
    elif species_code == 'BA2A':
        label_tensor[11] = 1  # SpermWhale
    else:
        print("Label not found")
        print(label)
    return label_tensor

# Example data and targets
img_dir = "../data/mfccData/train/images"

labels = []
# Create the dataset
for filename in os.listdir(img_dir):
        # Check if the item in the folder is a file (not a directory)
        if os.path.isfile(os.path.join(img_dir, filename)):
            species = filename.split("_")[0]
            species_code = filename.split("_")[1]

            if(species == "whale" and species_code != 'CC5A'):
                labels.append(filename)


labels = torch.stack([label_mapping(label) for label in labels])

weightTensor = []
countTensor =  [0] * 12
total = 0 


for idx in range(len(labels)):
    countTensor[torch.nonzero(labels[idx])[0]] = countTensor[torch.nonzero(labels[idx])[0]] + 1
    total = total + 1

for count in countTensor:
    weightTensor.append(round(len(labels)/count,3))


print("Counts")
print(countTensor)
print("Total:")
print(total)
print("Weights:")
print(weightTensor)