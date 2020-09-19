from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path

TEST_IMAGE_PATH = Path('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/data/processed/3rd-ml-month-car-image-cropping-dataset/test_crop/')

class TestDataset(Dataset):
    def __init__(self, df, mode='test', transforms=None, TEST_IMAGE_PATH=None):
        self.df = df
        self.mode = mode
        self.transform = transforms[self.mode]
        self.TEST_IMAGE_PATH = TEST_IMAGE_PATH

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = Image.open(TEST_IMAGE_PATH / self.df['img_file'][idx]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image