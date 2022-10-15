# Author(s): Justice Mason
# Project: DEVS

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DEVSdataset(Dataset):
    """
    Dataset class for the DEVS project.
    
    ...
    
    Attributes
    ----------
    data : torch.Tensor
        N-D array of images for training/testing.
        
    seq_len : int, default=3
        Number of observations representating a sequence of images -- input to the network.
        
        
    Methods
    -------
    __len__()
    __getitem__()
    
    Notes
    -----
    
    """
    def __init__(self, data: np.ndarray, seq_len: int = 3):
        super(DEVSdataset, self).__init__()
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        """
        """
        num_traj, traj_len, _, _, _ = self.data.shape
        length = num_traj * (traj_len - self.seq_len + 1)
            
        return length
        
    def __getitem__(self, idx):
        """
        """
        assert idx < self.__len__(),  "Index is out of range."
        num_traj, traj_len, _, _, _ = self.data.shape
        
        traj_idx, seq_idx = divmod(idx, traj_len - self.seq_len + 1)
        
        sample = self.data[traj_idx, seq_idx:seq_idx+self.seq_len,...]
        
        return sample

    
class ImageDataset(Dataset):
    """
    Generic image dataset.
    
    ...
    
    """
    def __init__(self, data: np.ndarray, traj_len: int = 15):
        super(ImageDataset, self).__init__()
        self.data = data
        self.traj_len = traj_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        assert idx < self.__len__(), "Index is out of range. Length of dataset: {}".format(self.__len__())
        traj_idx, seq_idx = divmod(idx, self.traj_len)
        
        return self.data[traj_idx, seq_idx, ...]