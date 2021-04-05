import cv2
import os
import random
import numpy as np 
import pandas as pd

import torch
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt 


class dataset(Dataset):
    """
    Custom Dataset 
    """
    def __init__(self, csv, img_path, transform=None):
        self.labels_df = pd.read_csv(csv)
        self.img_path = img_path
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self,index):
        image_path = os.path.join(self.img_path, self.labels_df.iloc[index,1])
        img = Image.open(image_path)
        
        y_label = torch.tensor(int(self.labels_df.iloc[index, 3]))
        
        if self.transform:
            img = self.transform(img)
            
        return [img, y_label]