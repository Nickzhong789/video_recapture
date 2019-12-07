from video_proc import *
import pandas as pd


# path = './video_data'
# generate_anno(path)

df_v = pd.read_csv('./video_anno.csv')
data_dir = '/media/ubuntu/1/data'
ratio = 0.8

data_list = []
for i in range(len(df_v)):
    video_info_list = df_v.iloc[i]
    isTrain = True if i < len(df_v) * ratio else False
    block_list = video_cap(video_info_list, data_dir, train=isTrain)
    for block in block_list:
        data_list.append(block)
    df = pd.DataFrame(data=data_list, columns=['block', 'label', 'index', 'isTrain'])
    df.to_csv('./anno.csv', index=None)
