import os
import torch
import cv2
from torch.utils.data import Dataset
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
        
        labels = []
        # Iterate through files in the folder
        for filename in os.listdir(img_dir):
            # Check if the item in the folder is a file (not a directory)
            if os.path.isfile(os.path.join(img_dir, filename)):
                labels.append(filename)

        self.img_labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
        species = self.img_labels[idx].split("_")[0]
        if(species == "orca"):
                label = torch.tensor([1, 0], dtype=torch.float32)
        elif(species == "dolphin"):
                label = torch.tensor([0, 1], dtype=torch.float32)
        else:
            print("label not found")

        if self.transform:
            try:
                image = self.transform(image)
            except:
                print(img_path)
        return image, label

    def __next__(self):
        if self.num >= self.max:
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num - 1)


