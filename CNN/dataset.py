import os
import torch
import cv2
from torch.utils.data import Dataset
from torch import tensor
from torchvision import transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #AddGaussianNoise(0., 0)
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
        self.fileName = []
        # Iterate through files in the folder
        
        for filename in os.listdir(img_dir):
            # Check if the item in the folder is a file (not a directory)
            if os.path.isfile(os.path.join(img_dir, filename)):
                species = filename.split("_")[0]
                species_code = filename.split("_")[1]
                
                if((species == "whale" and species_code != 'CC5A')):
                   self.labels.append(filename)
                   self.fileName.append(filename)

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
        return self.data[idx], self.labels[idx], self.fileName[idx]

    def __next__(self):
        if self.num >= self.max:
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num - 1)
        
    def label_mapping(self, label):
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
    
    # TODO change to have the 4 species 
    #def label_mapping(self, label):
    #    species_code = label.split("_")[0]
    #    label_tensor = torch.zeros(4, dtype=torch.float32)
     #   if species_code == 'dolphin':
      #      label_tensor[0] = 1  # dolphin
       # elif species_code == 'orca':
       #     label_tensor[1] = 1  # orca
       # elif species_code == 'seal':
        #    label_tensor[2] = 1  # seal
        #elif species_code == 'whale':
       #     label_tensor[3] = 1  # whale
       # else:
        ##    print("Label not found")
        #    print(label)
       # return label_tensor

