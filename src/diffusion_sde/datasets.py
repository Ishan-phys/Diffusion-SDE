from PIL import Image
import os
from diffusion_sde.configs.config import CFGS
from torch.utils.data import Dataset

class datasets(Dataset):
    def __init__(self, root_img):
        self.root_img = root_img
        self.transform = None
        self.images = os.listdir(root_img)
        self.length_dataset = len(self.images)
        
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        img = self.images[index % self.length_dataset]
    
        img_path = os.path.join(self.root_img, img)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        return img      
    
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.subset)