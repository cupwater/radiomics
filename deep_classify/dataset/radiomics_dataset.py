'''
Author: Peng Bo
Date: 2022-06-08 13:28:30
LastEditTime: 2022-08-27 18:24:21
Description: 

'''
import os
import numpy as np

import torch
from torch.utils.data import Dataset


__all__ = ['RadiomicsDataset']

class RadiomicsDataset(Dataset):
    def __init__(self, features_path, metas_path):
        self.features = np.loadtxt(features_path).astype(np.float32)
        self.metas    = np.loadtxt(metas_path).astype(np.int32)

    def __getitem__(self, index):
        feat  = torch.FloatTensor(torch.from_numpy(self.features[index]))
        label = torch.tensor(self.metas[index], dtype=torch.long)
        return feat, label

    def __len__(self):
        return self.metas.shape[0]

    def feature_dim(self):
        return self.features.shape[1]
    
    def num_classes(self):
        return np.max(self.metas) + 1

if __name__ == "__main__":
    train_list = '../radiomics_data/data1/RA-C.feat'
    train_meta = '../radiomics_data/data1/RA-C.meta'

    trainset = RadiomicsDataset(train_list, train_meta)
    
    import pdb
    pdb.set_trace()
    trainset.__getitem__(1)