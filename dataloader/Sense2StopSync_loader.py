import itertools
import numpy as np
import os
import pickle
import pandas as pd
import random
import sys
import time

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from S2S_segment import segment_all
from S2S_settings import settings


class S2S_dataset(Dataset):
    def __init__(self, settings, usage):
        self.df_dataset, self.info_dataset = segment_all(
            window_size_sec=settings["window_size_sec"], 
            stride_sec=settings["stride_sec"], 
            window_criterion=settings["window_criterion"],
            usage=usage
        )
        self.length = len(self.df_dataset)
        
    def __getitem__(self, index):
        data = self.df_dataset[index]
        info = self.info_dataset[index]
        print(data)
        sense_feat = data[['accx', 'accy', 'accz']].to_numpy() # 'accx', 'accy', 'accz', 'acc_pca', 'flowx', 'flowy', 'diff_flowx', 'diff_flowy', 'diffflow_pca'
        print(sense_feat.type)
        # sense_feat = sense_feat.transpose([1, 0]).astype(np.float32)
        # flow_feat = data[['diffflow_pca']].to_numpy()
        # flow_feat = flow_feat.transpose([1, 0]).astype(np.float32)
        return sense_feat, info
    
    def __len__(self):
        return self.length


if __name__ == "__main__":
    train_dataset = S2S_dataset(settings=settings, train=True)
    print(train_dataset[0])
    for i in range(len(train_dataset)):
        flow_feat, sense_feat, info = train_dataset[i]
        if flow_feat.shape != (1, 300):
            print('flow', flow_feat.shape)
        if sense_feat.shape != (1, 300):
            print('sense', sense_feat.shape)

