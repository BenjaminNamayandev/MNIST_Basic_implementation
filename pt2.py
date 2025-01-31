import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd

class SportsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform)

    def __len__(self):
        return len(self.data)

    def __getitem(self, idx): 
        return self.data[idx]
    
    def classes(self):
        return self.data.classes
    

dataset = SportsDataset('./dataset/train') # 100 classes
