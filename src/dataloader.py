import torch
import numpy as np
import pandas as pd

class KaggleMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str):
        raw_data = pd.read_csv(file_path)
        self.raw_shape = raw_data.shape
        
        
        if self.raw_shape[1] == 785:
            self.train = True
            self.y = raw_data.values[:, 0]
            self.x = raw_data.values[:, 1 :] / 255
            
        elif self.raw_shape[1] == 784:
            self.train = False
            self.x = np.array(raw_data) / 255
            
        self.x = np.reshape(self.x, (self.x.shape[0], 28, 28))
            
    
    def __len__(self):
        return self.raw_shape[0]
    
    def __getitem__(self, index):
        image = self.x[index]

        if self.train:
            label = self.y[index]
            return torch.Tensor(image), label

        else:
            return torch.Tensor(image)