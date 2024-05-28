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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example data and targets
data_test = "../data/mfccData/test/images"
model_path = "./model_12Whales.pth"

if os.path.exists(model_path):
    print("Model path exists.")

# Create the dataset
dataset_test = Marine_Mammal_Dataset(data_test)

# Instantiations
model = CNN(dataset_test[0][0].shape)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

guessTensor = [0] * 12
correctTensor = [0] * 12
correct = 0

# Initialize lists to store predictions and ground truth labels
predictions = []
ground_truths = []

total_execution_time = 0

for idx in range(len(dataset_test)):
    input = dataset_test[idx][0].unsqueeze(0)
    start_time = time.time()
    input = input.to(device)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time
    output = model(input)[0]
    value, indices = torch.max(output, 0)

    # Update guessTensor and correctTensor
    guessTensor[torch.nonzero(dataset_test[idx][1])[0]] += 1
    
    if dataset_test[idx][1][indices] == 1.:
        correctTensor[indices] += 1
        correct += 1

    predictions.append(indices.item())
    ground_truths.append(torch.nonzero(dataset_test[idx][1]).item())



average_execution_time = total_execution_time / len(dataset_test)
print("Average execution time of 'input = input.to(device)':", average_execution_time, "seconds")

# Convert lists to numpy arrays
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

# Calculate confusion matrix
conf_matrix = confusion_matrix(ground_truths, predictions)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(14, 10))
sns.set(font_scale=1.4)  # Adjust font size for better readability
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=.5)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig("./confusion_matrix_4animals.png")

# Calculate overall accuracy
accuracy = correct / len(dataset_test)
print('Overall Accuracy:')
print(accuracy)

print('Guess Distribution:')
print(guessTensor)

print('Correct Guess Distribution:')
print(correctTensor)