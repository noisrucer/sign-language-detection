import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import os.path as osp


class CustomDataset(Dataset):
    def __init__(self, data_dir='./data'):
        super().__init__()
        self.data_dir = data_dir
        self.fnames = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.n_samples = len(self.fnames)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = osp.join(self.data_dir, fname)

        # Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return 0




