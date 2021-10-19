import torch
import torch.nn as nn
import numpy as np
from torch.utils import data 

class Dataset(data.Dataset): 
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        X = self.data[idx][:, :12, :]
        y = self.data[idx][:, 12:, :]
        return X, y