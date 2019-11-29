from torch.utils.data import Dataset
import pandas as pd
from os.path import join

from video_proc import *


class VideoDataset(Dataset):
    def __init__(self, path, anno, start_train, train_num, start_test, test_num, train=True):
        self.df = pd.read_csv(join(path, anno))
        self.path = path
        self.train = train

        self.start_train = start_train
        self.train_num = train_num
        self.start_test = start_test
        self.test_num = test_num

        videoTrainList = self.df[start_train:start_train + train_num]
        videoTestList = self.df[start_test:start_test + test_num]

        self.blockTrainList = get_blocks(videoTrainList)
        self.blockTestList = get_blocks(videoTestList)
        self.blockTrainNum = len(self.blockTrainList)
        self.blockTestNum = len(self.blockTestList)

        print('blockTrainNum: ', self.blockTrainNum)
        print('blockTestNum: ', self.blockTestNum)

    def __getitem__(self, idx):
        if self.train:
            block, block_label, block_idx = self.blockTrainList[idx]
        else:
            block, block_label, block_idx = self.blockTestList[idx]

        return block, block_label, block_idx

    def __len__(self):
        if self.train:
            return self.blockTrainNum
        else:
            return self.blockTestNum
