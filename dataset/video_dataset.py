from torch.utils.data import Dataset
import pandas as pd
from os.path import join

from video_proc import *


class VideoDataset(Dataset):
    def __init__(self, path, anno, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = transform
        self.df = pd.read_csv(join(path, anno))
        self.path = path
        self.train = train

        df_v = self.df[:3]

        self.length = len(df_v)
        segId = int(self.length * 0.8)
        videoTrainList = df_v[:segId]
        videoTestList = df_v[segId:]

        self.frameTrainList = []
        self.frameTestList = []
        self.blockTrainNum = 0
        self.blockTestNum = 0

        for i in range(len(videoTrainList)):
            f_list, f_num = get_frame(videoTrainList.iloc[i])
            self.frameTrainList.append(frame for frame in f_list)  # [[frame, f_label, f_id]]
            self.blockTrainNum += f_num * 100

        for i in range(len(videoTestList)):
            f_list, f_num = get_frame(videoTestList.iloc[i])
            self.frameTestList.append(frame for frame in f_list)
            self.blockTestNum += f_num * 100

        print('blockTrainNum: ', self.blockTrainNum)
        print('blockTestNum: ', self.blockTestNum)

    def __getitem__(self, idx):
        if self.train:

            block, block_label, block_id = self.blockTrainList[idx]
        else:
            block, block_label, block_id = self.blockTestList[idx]

        return block, block_label, block_id

    def __len__(self):
        if self.train:
            return len(self.blockTrainList)
        else:
            return len(self.blockTestList)
