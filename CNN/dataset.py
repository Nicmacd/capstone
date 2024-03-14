import os
import torch
import cv2
from torch.utils.data import Dataset
from torch import tensor
from torchvision import transforms

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

label_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

class Marine_Mammal_Dataset(Dataset):
    def __init__(
        self, img_dir, transform=data_transform, target_transform=label_transfrom
    ):
        
        self.data = []
        self.labels = []
        # Iterate through files in the folder
        for filename in os.listdir(img_dir):
            # Check if the item in the folder is a file (not a directory)
            if os.path.isfile(os.path.join(img_dir, filename)):
                self.labels.append(filename)

        for label in self.labels:
            print("Loading image for label: " + label)
            img_path = os.path.join(img_dir, label)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = transform(image)
            self.data.append(image)
 
        self.data = torch.stack(self.data)
        self.labels = torch.stack([self.label_mapping(label) for label in self.labels])

        self.data = self.data.to(torch.device("cuda"))
        self.labels = self.labels.to(torch.device("cuda"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __next__(self):
        if self.num >= self.max:
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num - 1)
        
    def label_mapping(self, label):
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

