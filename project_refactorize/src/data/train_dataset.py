# library import
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path

TRAIN_IMAGE_PATH = Path('/home/sykim/Desktop/project_refactorize/data/processed/3rd-ml-month-car-image-cropping-dataset/train_crop/')

class TrainDataset(Dataset):
    def __init__(self, df, mode='train', transforms=None):
        self.df = df
        self.mode = mode
        self.transform = transforms[self.mode]     
        
    def __len__(self):
        return len(self.df)
            
    def __getitem__(self, idx):
        image = Image.open(TRAIN_IMAGE_PATH / self.df['img_file'][idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.df['class'][idx]

        return image, label

'''
class TrainDataset(Dataset):
    def __init__(self, df, mode='train', transforms=None):
        self.df = df
        self.mode = mode
        self.transform = transforms[self.mode]
        
    def __len__(self):
        return len(self.df)
            
    def __getitem__(self, idx):
        
        image = Image.open(TRAIN_IMAGE_PATH / self.df['img_file'][idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.df['class'][idx]

        return image, label
'''