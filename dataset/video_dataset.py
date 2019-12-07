from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from os.path import join

from video_proc import *


class VideoDataset(Dataset):
    def __init__(self, path, anno, train=True):
        self.df = pd.read_csv(join(path, anno))
        self.path = path
        self.train = train

        self.blockTrainDf = self.df[self.df['isTrain'] == 'train']
        self.blockTestDf = self.df[self.df['isTrain'] == 'test']
        self.blockTrainNum = len(self.blockTrainDf)
        self.blockTestNum = len(self.blockTestDf)

        print('blockTrainNum: ', self.blockTrainNum)
        print('blockTestNum: ', self.blockTestNum)

    def __getitem__(self, idx):
        if self.train:
            block_dir, block_label, block_idx, isTrain = self.blockTrainDf.iloc[idx]
            block = np.load(block_dir)
        else:
            block_dir, block_label, block_idx, isTrain = self.blockTestDf.iloc[idx]
            block = np.load(block_dir)

        return block, block_label, block_idx

    def __len__(self):
        if self.train:
            return self.blockTrainNum
        else:
            return self.blockTestNum
